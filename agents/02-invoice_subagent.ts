import "dotenv/config";
import { z } from "zod/v3";
import { createAgent, tool } from "langchain";
import { SqlDatabase } from "@langchain/classic/sql_db";
import { setupDatabase, AgentState, defaultModel } from "./utils.js";
import { getCurrentTaskInput } from "@langchain/langgraph";

// ============================================================================
// 在 TOOLS 內存取 STATE：強大的安全模式
// ============================================================================
//
// 此檔案示範一個關鍵模式：從 tools 內部存取 graph state。
// 
// 為什麼這很重要？
// 不要求 LLM 將敏感資料（例如 customer IDs）作為 tool 參數傳入，
// 而是直接從 graph state 讀取。這有幾個主要好處：
//
// 1. SECURITY：LLM 永遠不會看到或處理敏感識別碼
// 2. RELIABILITY：不會有 LLM 傳入錯誤或幻覺 ID 的風險
// 3. CLEANER：已存在於 state 的上下文，不需要再作為額外參數傳給 tools
//
// 關鍵函數：getCurrentTaskInput()
// 此函數讓你可以從任何地方存取目前 graph state，包括 tools 內部。
// 它就像所有 tools 都能讀取的「global context」。

// ============================================================================
// Tools
// ============================================================================

async function createInvoiceTools(db: SqlDatabase) {
  const getInvoicesByCustomerSortedByDate = tool(
    async () => {
      // 關鍵模式：從 graph state 取得 customerId，而不是從 LLM 取得
      // 這可確保 LLM 不會意外或惡意傳入錯誤的 customer ID
      const state = await getCurrentTaskInput<AgentState>();
      const customerId = state.customerId;

      if (!customerId) {
        return "Error: Customer ID not found in state. Customer must be verified first.";
      }

      // 現在可以安全使用已驗證的 customer ID
      const query = `SELECT * FROM Invoice WHERE CustomerId = ${customerId} ORDER BY InvoiceDate DESC;`;
      const rawResult = await db.run(query);
      const result =
        typeof rawResult === "string" ? JSON.parse(rawResult) : rawResult;
      return JSON.stringify(result);
    },
    {
      name: "get_invoices_by_customer_sorted_by_date",
      description:
        "Look up all invoices for the current customer. The invoices are sorted in descending order by invoice date. The customer ID is automatically retrieved from the state.",
      // 注意：schema 是空的，不需要參數！
      // LLM 只要用空參數呼叫此 tool
      schema: z.object({}),
    }
  );

  const getInvoicesSortedByUnitPrice = tool(
    async () => {
      // 相同模式：從 state 讀取 customer ID，而不是要求 LLM 提供
      const state = await getCurrentTaskInput<AgentState>();
      const customerId = state.customerId;

      if (!customerId) {
        return "Error: Customer ID not found in state. Customer must be verified first.";
      }

      const query = `
        SELECT Invoice.*, InvoiceLine.UnitPrice
        FROM Invoice
        JOIN InvoiceLine ON Invoice.InvoiceId = InvoiceLine.InvoiceId
        WHERE Invoice.CustomerId = ${customerId}
        ORDER BY InvoiceLine.UnitPrice DESC;
      `;
      const rawResult = await db.run(query);
      const result =
        typeof rawResult === "string" ? JSON.parse(rawResult) : rawResult;
      return JSON.stringify(result);
    },
    {
      name: "get_invoices_sorted_by_unit_price",
      description:
        "Use this tool when the customer wants to know the details of one of their invoices based on the unit price/cost. This tool looks up all invoices for the current customer and sorts by unit price. The customer ID is automatically retrieved from the state.",
      schema: z.object({}),
    }
  );

  const getEmployeeByInvoiceAndCustomer = tool(
    async ({ invoiceId }) => {
      // 此 tool 確實會從 LLM 取得一個參數 invoiceId
      // 但基於安全性，仍會從 state 讀取 customerId
      const state = await getCurrentTaskInput<AgentState>();
      const customerId = state.customerId;

      if (!customerId) {
        return "Error: Customer ID not found in state. Customer must be verified first.";
      }

      const query = `
        SELECT Employee.FirstName, Employee.Title, Employee.Email
        FROM Employee
        JOIN Customer ON Customer.SupportRepId = Employee.EmployeeId
        JOIN Invoice ON Invoice.CustomerId = Customer.CustomerId
        WHERE Invoice.InvoiceId = ${invoiceId} AND Invoice.CustomerId = ${customerId};
      `;
      const rawResult = await db.run(query);
      const result =
        typeof rawResult === "string" ? JSON.parse(rawResult) : rawResult;

      if (!result || result.length === 0) {
        return `No employee found for invoice ID ${invoiceId} and customer identifier ${customerId}.`;
      }
      return JSON.stringify(result);
    },
    {
      name: "get_employee_by_invoice_and_customer",
      description:
        "This tool will take in an invoice ID and return the employee information associated with the invoice. The customer ID is automatically retrieved from the state.",
      schema: z.object({
        invoiceId: z.number().describe("The ID of the specific invoice"),
      }),
    }
  );

  return [
    getInvoicesByCustomerSortedByDate,
    getInvoicesSortedByUnitPrice,
    getEmployeeByInvoiceAndCustomer,
  ];
}

// ============================================================================
// System Prompt
// ============================================================================
//
// system prompt 會引導此專門 subagent 的行為。
// 注意它會告訴 LLM：customer ID 會自動處理。

const invoiceSubagentPrompt = `
<important_background>
You are a subagent among a team of assistants. You are specialized for retrieving and processing invoice information. 
Invoices contain information such as song purchases and billing history. Only respond to questions if they relate in some way to billing, invoices, or purchases.  
If you are unable to retrieve the invoice information, respond that you are unable to retrieve the information.
IMPORTANT: Your interaction with the customer is done through an automated system. You are not directly interacting with the customer, so avoid chitchat or follow up questions and focus PURELY on responding to the request with the necessary information. 
</important_background>
 
<tools>
You have access to three tools. These tools enable you to retrieve and process invoice information from the database. Here are the tools:
- get_invoices_by_customer_sorted_by_date: Retrieves all invoices for the current customer (no parameters needed - customer ID is automatically retrieved from state)
- get_invoices_sorted_by_unit_price: Retrieves all invoices for the current customer sorted by unit price (no parameters needed - customer ID is automatically retrieved from state)
- get_employee_by_invoice_and_customer: Retrieves employee information for a specific invoice (only requires invoiceId - customer ID is automatically retrieved from state)

IMPORTANT: The customer ID is automatically retrieved from the graph state, so you don't need to pass it as a parameter. The customer must be verified before these tools can be used.
</tools>

<core_responsibilities>
- Retrieve and process invoice information from the database
- Provide detailed information about invoices, including customer details, invoice dates, total amounts, employees associated with the invoice, etc. when the customer asks for it.
- Always maintain a professional, friendly, and patient demeanor in your responses.
</core_responsibilities>

You may have additional context that you should use to help answer the customer's query. It will be provided to you below:
`;

// ============================================================================
// Agent 建立
// ============================================================================

console.log("💰 Creating Invoice Information Subagent...");

// 設定資料庫
const db = await setupDatabase();

// 建立 tools
const invoiceTools = await createInvoiceTools(db);

// 使用 shared state schema 建立 agent
const agent = createAgent({
  model: defaultModel,
  tools: invoiceTools,
  systemPrompt: invoiceSubagentPrompt,
  
  // 關鍵：stateSchema 參數
  // 提供 AgentState 後，tools 就能透過 getCurrentTaskInput() 存取 state
  // 這會把 agent 的 state 連接到 tools，讓它們可以讀取 customerId
  // 沒有這個設定，getCurrentTaskInput() 就無法運作！
  stateSchema: AgentState,
});

console.log("✅ Invoice Information Subagent created successfully!");

// ============================================================================
// 匯出
// ============================================================================
//
// 此 agent 會被匯入，並在 supervisor workflow 中作為 tool 使用。
// supervisor 會在 state 中傳入 customerId，而此 agent 的 tools
// 會透過 getCurrentTaskInput() 自動存取它。

export const graph = agent.graph;
