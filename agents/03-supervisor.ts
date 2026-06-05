import "dotenv/config";
import { z } from "zod/v3";
import { createAgent, tool, HumanMessage } from "langchain";
import {
  MemorySaver,
  InMemoryStore,
  getCurrentTaskInput,
} from "@langchain/langgraph";
import { graph as musicCatalogSubagent } from "./01-music_subagent.js";
import { graph as invoiceInformationSubagent } from "./02-invoice_subagent.js";
import { AgentState, defaultModel } from "./utils.js";

// ============================================================================
// 將 SUBAGENTS 作為 TOOLS：建議模式
// ============================================================================
//
// 此檔案示範一個強大且清楚的架構模式：
// 將 subagents 包裝成 supervisor 可以呼叫的 tools。
//
// 為什麼使用此模式？
// - MODULARITY：每個 subagent 都是獨立的，可分開開發與測試
// - SIMPLICITY：supervisor 只需像使用其他 tool 一樣委派給專門 subagents
// - REUSABILITY：subagents 可在不同 workflows 中重複使用
// - SCALABILITY：容易加入新的專門 subagents
//
// 運作方式：
// 1. 匯入已編譯的 subagent graphs（來自 01 和 02）
// 2. 將每個 subagent 包裝成 tool function
// 3. tool 呼叫 subagent graph 並回傳它的 response
// 4. Supervisor 使用 createAgent() 並傳入這些 subagent tools
//
// 與其他模式比較：
// - 比傳統 hierarchical supervisor patterns 更簡單
// - 比 monolithic agents 更有彈性
// - 比 "swarm" approaches 有更好的關注點分離

// ============================================================================
// System Prompt
// ============================================================================

const supervisorPrompt = `
<background>
You are an expert customer support assistant for a digital music store. You can handle music catalog or invoice related question regarding past purchases, song or album availabilities. 
You are dedicated to providing exceptional service and ensuring customer queries are answered thoroughly, and have a team of subagents that you can use to help answer queries from customers. 
Your primary role is to delegate tasks to this multi-agent team in order to answer queries from customers. 
</background>

<important_instructions>
Always respond to the customer through summarizing the findings of the individual responses from subagents. 
If a question is unrelated to music or invoice, politely remind the customer regarding your scope of work. Do not answer unrelated answers.
Based on the existing steps that have been taken in the messages, your role is to call the appropriate subagent based on the users query.
</important_instructions>

<tools>
You have 2 tools available to delegate to the subagents on your team:
1. music_catalog_subagent: Call this tool to delegate to the music subagent. The music agent has access to user's saved music preferences. It can also retrieve information about the digital music store's music 
catalog (albums, tracks, songs, etc.) from the database. 
2. invoice_information_subagent: Call this tool to delegate to the invoice subagent. This subagent is able to retrieve information about a customer's past purchases or invoices 
from the database. The customer ID is automatically retrieved from the state, so you don't need to pass it.
</tools>
`;

// ============================================================================
// Supervisor Tools：包裝 Subagents
// ============================================================================
//
// 這裡會將每個 subagent graph 包裝成 tool。
// supervisor 會把它們視為一般 tools，但它們實際上
// 會呼叫完整的 agent workflows！

// INVOICE SUBAGENT TOOL
// 此 tool 會包裝 invoice subagent，並處理 state 傳遞
const callInvoiceInformationSubagent = tool(
  async ({ query }) => {
    // 從 supervisor 的 state 取得 customerId
    // 我們使用 getCurrentTaskInput() 存取它，這與 02-invoice_subagent.ts 的模式相同
    const state = await getCurrentTaskInput<AgentState>();

    // 使用 query 和 state 呼叫 invoice subagent graph
    // customerId 會透過 state 傳遞，讓 subagent 的 tools 可以存取它
    const result: any = await invoiceInformationSubagent.invoke({
      messages: [new HumanMessage(query)],
      customerId: state.customerId,  // 透過 state 傳遞 context
    });
    
    // 取出 subagent 的最終 response
    const subagentResponse = result.messages.at(-1).content;
    return subagentResponse;
  },
  {
    name: "invoice_information_subagent",
    description:
      "An agent that can assist with all invoice-related queries. It can retrieve information about a customer's past purchases or invoices. The customer ID is automatically retrieved from the state.",
    schema: z.object({
      query: z.string().describe("The query to send to the invoice subagent"),
    }),
  }
);

// MUSIC CATALOG SUBAGENT TOOL
// 此 tool 會包裝 music catalog subagent
const callMusicCatalogSubagent = tool(
  async ({ query }) => {
    // 呼叫 music catalog subagent graph
    // 此 subagent 不需要 customerId，所以只傳入 query
    const result: any = await musicCatalogSubagent.invoke({
      messages: [new HumanMessage(query)],
    });
    
    // 取出並回傳 subagent 的 response
    const subagentResponse = result.messages.at(-1).content;
    return subagentResponse;
  },
  {
    name: "music_catalog_subagent",
    description:
      "An agent that can assist with all music-related queries. This agent has access to user's saved music preferences. It can also retrieve information about the digital music store's music catalog (albums, tracks, songs, etc.) from the database.",
    schema: z.object({
      query: z
        .string()
        .describe("The query to send to the music catalog subagent"),
    }),
  }
);

// ============================================================================
// Agent 建立
// ============================================================================
//
// SUPERVISOR AGENT
// supervisor 只是用 createAgent() 建立的一般 agent。
// 它特別之處在於它的 "tools" 實際上是其他 agents！
//
// 架構優點：
// - supervisor 專注於 routing/delegation
// - 每個 subagent 都是 domain expert
// - 清楚分離關注點
// - 易於測試與維護

console.log("👔 Creating Supervisor Agent...");

// 初始化 memory stores，用於 conversation persistence
const checkpointer = new MemorySaver();
const inMemoryStore = new InMemoryStore();

// 建立 supervisor agent
export const supervisor = createAgent({
  model: defaultModel,
  
  // 這些 "tools" 實際上是 subagent wrappers！
  // 對 LLM 來說，它們看起來像一般 tools
  tools: [callInvoiceInformationSubagent, callMusicCatalogSubagent],
  
  systemPrompt: supervisorPrompt,
  
  // stateSchema 讓 tool wrappers 中可以使用 getCurrentTaskInput()
  stateSchema: AgentState,
  
  checkpointer,
  store: inMemoryStore,
});

console.log("✅ Supervisor Agent created successfully!");

// ============================================================================
// 匯出
// ============================================================================
//
// 此 supervisor 可以單獨使用，也可以作為較大型 workflow 的一部分
// 例如 04-supervisor_with_verification.ts

export const graph = supervisor.graph;
