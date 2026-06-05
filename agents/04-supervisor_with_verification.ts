import "dotenv/config";
import { z } from "zod/v3";
import { BaseMessage, HumanMessage, SystemMessage, AIMessage } from "langchain";
import {
  MessagesZodMeta,
  StateGraph,
  START,
  END,
  MemorySaver,
  InMemoryStore,
  interrupt,
} from "@langchain/langgraph";
import { withLangGraph } from "@langchain/langgraph/zod";
import { SqlDatabase } from "@langchain/classic/sql_db";
import { setupDatabase, defaultModel, AgentState } from "./utils.js";
import { supervisor } from "./03-supervisor.js";

// ============================================================================
// 搭配 CUSTOMER VERIFICATION 的 HUMAN-IN-THE-LOOP
// ============================================================================
//
// 此檔案以 supervisor pattern（03）為基礎，加入 verification 步驟。
// supervisor 協助處理 invoices 前，customer 必須先驗證身分。
//
// 核心概念：
// - Human-in-the-loop：暫停執行以取得 user input
// - interrupt()：LangGraph 中用來暫停並等待 input 的函數
// - 依據 verification state 進行 conditional routing
// - 使用 structured output parsing 擷取 customer information
//
// WORKFLOW：
// 1. User 送出 query
// 2. verify_info node 檢查 customer 是否已驗證
// 3. 如果尚未驗證，使用 interrupt() 要求 credentials
// 4. 如果已驗證，繼續到 supervisor
//
// 此模式適用於：
// - Authentication/authorization
// - Confirmation prompts
// - 收集必要資訊
// - Approval workflows

// ============================================================================
// State 定義
// ============================================================================

// 使用 Zod 定義 Input State
// 這會限制呼叫 graph 時可提供的欄位
const InputStateAnnotation = z.object({
  messages: withLangGraph(z.custom<BaseMessage[]>(), MessagesZodMeta),
});

// ============================================================================
// Customer Verification Helpers（客戶驗證輔助函數）
// ============================================================================
//
// 這些 helper functions 會處理 customer verification logic。
// 它們可以用 ID、email 或 phone number 查找 customers。

// 輔助函數：從各種 identifiers 查找 customer ID
async function getCustomerIdFromIdentifier(
  identifier: string,
  db: SqlDatabase
): Promise<number | null> {
  // 直接 customer ID（數字）
  if (/^\d+$/.test(identifier)) {
    return parseInt(identifier);
  }

  // Phone number 查找
  if (identifier.startsWith("+")) {
    // 移除空格與括號來正規化，以便彈性比對
    const normalizedInput = identifier.replace(/[\s\(\)]/g, "");

    // 先嘗試 exact match
    const query = `SELECT CustomerId FROM Customer WHERE Phone = '${identifier}';`;
    const rawResult = await db.run(query);
    const result =
      typeof rawResult === "string" ? JSON.parse(rawResult) : rawResult;

    if (result && result.length > 0) {
      return result[0].CustomerId;
    }

    // 如果 exact match 失敗，嘗試 normalized match
    const queryAll = `SELECT CustomerId, Phone FROM Customer WHERE Phone LIKE '+%';`;
    const rawAllPhones = await db.run(queryAll);
    const allPhones =
      typeof rawAllPhones === "string"
        ? JSON.parse(rawAllPhones)
        : rawAllPhones;

    for (const row of allPhones) {
      if (row.Phone) {
        const normalizedDb = row.Phone.replace(/[\s\(\)]/g, "");
        if (normalizedDb === normalizedInput) {
          return row.CustomerId;
        }
      }
    }
  }

  // Email 查找
  if (identifier.includes("@")) {
    const query = `SELECT CustomerId FROM Customer WHERE Email = '${identifier}';`;
    const rawResult = await db.run(query);
    const result =
      typeof rawResult === "string" ? JSON.parse(rawResult) : rawResult;

    if (result && result.length > 0) {
      return result[0].CustomerId;
    }
  }

  return null;
}

// ============================================================================
// System Prompts
// ============================================================================

// ============================================================================
// Schemas
// ============================================================================
//
// STRUCTURED OUTPUT PARSING
// 我們使用 Zod schemas 搭配 withStructuredOutput() 從 user messages 擷取特定資訊。
// 這比 regex 或 string parsing 更可靠。

// 用於解析 user-provided account information 的 schema
const UserInputSchema = z.object({
  identifier: z
    .string()
    .describe(
      "Identifier, which can be a customer ID, email, or phone number."
    ),
});

// ============================================================================
// Nodes
// ============================================================================
//
// 此 graph 中的 nodes 會實作 verification workflow。

// ============================================================================
// Conditional Edge
// ============================================================================
//
// 用於 VERIFICATION 的 CONDITIONAL ROUTING
// 此函數會決定要繼續到 supervisor，或是中斷以進行 verification。

function shouldInterrupt(state: AgentState): "continue" | "interrupt" {
  // 如果 state 中存在 customerId，代表 customer 已驗證
  if (state.customerId !== undefined) {
    return "continue";  // 前往 supervisor
  } else {
    return "interrupt";  // 需要收集 credentials
  }
}

// ============================================================================
// Graph 建立
// ============================================================================

console.log("👔 Creating Supervisor with Verification...");

// 設定資料庫
const db = await setupDatabase();

// 建立 structured output model，用於擷取 customer identifier
// withStructuredOutput() 會讓 LLM 回傳符合 schema 的資料
const structuredLlm = defaultModel.withStructuredOutput(UserInputSchema);

const structuredSystemPrompt = `You are a customer service representative responsible for extracting customer identifier.
Only extract the customer's account information from the message history. 
If they haven't provided the information yet, return an empty string for the identifier`;

// VERIFY INFO NODE
// 此 node 會嘗試擷取並驗證 customer identity
async function verifyInfo(state: AgentState) {
  if (state.customerId === undefined) {
    const systemInstructions = `
You are a music store agent, where you are trying to verify the customer identity as the first step of the customer support process. 
You cannot support them until their account is verified. 
In order to verify their identity, one of their customer ID, email, or phone number needs to be provided.
If the customer has not provided their identifier, please ask them for it.
If they have provided the identifier but cannot be found, please ask them to revise it.

IMPORTANT: Do NOT ask any questions about their request, or make any attempt at addressing their request until their identity is verified. It is CRITICAL that you only ask about their identity for security purposes.
`;

    const userInput = state.messages[state.messages.length - 1];

    // 使用 structured output 從 user message 擷取 identifier
    const parsedInfo = await structuredLlm.invoke([
      new SystemMessage(structuredSystemPrompt),
      userInput,
    ]);

    // 使用擷取出的 identifier 嘗試查找 customer
    const customerId = parsedInfo.identifier
      ? await getCustomerIdFromIdentifier(parsedInfo.identifier, db)
      : null;

    if (customerId !== null) {
      // 成功！Customer 已驗證
      const intentMessage = new AIMessage(
        `Thank you for providing your information! I was able to verify your account with customer id ${customerId}.`
      );
      return {
        customerId: customerId,
        messages: [intentMessage],
      };
    } else {
      // 無法驗證，要求 credentials 或 clarification
      const response = await defaultModel.invoke([
        new SystemMessage(systemInstructions),
        ...state.messages,
      ]);
      return { messages: [response] };
    }
  } else {
    // Customer ID 已存在於 state，代表已經驗證過
    return {};
  }
}

// HUMAN INPUT NODE
// 此 node 使用 interrupt() 暫停執行並收集 user input
function humanInput() {
  // interrupt() 會暫停 graph，並將控制權交還給 caller
  // caller 必須提供 input 才能恢復執行
  const userInput = interrupt("Please provide input.");
  return { messages: [new HumanMessage(userInput)] };
}

// SUPERVISOR NODE
// 簡單 wrapper，用來呼叫 03-supervisor.ts 中的 supervisor
async function supervisorNode(state: AgentState) {
  const result = await supervisor.invoke({
    ...state,
    customerId: state.customerId,
  });
  return {
    messages: result.messages,
  };
}

// 初始化 memory stores
const checkpointer = new MemorySaver();
const inMemoryStore = new InMemoryStore();

// 建立包含 human-in-the-loop verification 的 graph
const multiAgentVerify = new StateGraph(AgentState, {
  input: InputStateAnnotation,  // 限制呼叫時可傳入的內容
})
  .addNode("verify_info", verifyInfo)
  .addNode("human_input", humanInput)
  .addNode("supervisor", supervisorNode)
  
  // 從 verification 開始
  .addEdge(START, "verify_info")
  
  // 根據 verification status 進行路由
  .addConditionalEdges("verify_info", shouldInterrupt, {
    continue: "supervisor",      // Customer verified → proceed
    interrupt: "human_input",    // Need credentials → interrupt
  })
  
  // human input 後，再次嘗試 verification
  .addEdge("human_input", "verify_info")
  
  // supervisor 回覆後即完成
  .addEdge("supervisor", END);

// 編譯並匯出 graph
export const graph = multiAgentVerify.compile({
  checkpointer,        // interrupt() 運作所需
  store: inMemoryStore,
});

console.log("✅ Supervisor with Verification created successfully!");
