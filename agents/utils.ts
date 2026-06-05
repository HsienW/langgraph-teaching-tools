import { SqlDatabase } from "@langchain/classic/sql_db";
import { DataSource } from "typeorm";
import initSqlJs from "sql.js";
import { z } from "zod/v3";
import { initChatModel, BaseMessage } from "langchain";
import { MessagesZodMeta } from "@langchain/langgraph";
import { withLangGraph } from "@langchain/langgraph/zod";

// ============================================================================
// 所有 AGENTS 共用的 UTILITIES
// ============================================================================
//
// 此檔案包含所有 agent 範例共用的資源：
// - Model initialization
// - State schema definitions
// - Database setup
//
// 將這些集中管理可減少重複，並確保一致性。

// ============================================================================
// Model Initialization
// ============================================================================
//
// 使用 initChatModel()
// 這是 LangChain 的通用 chat model initializer。
// 它可使用簡單的 "provider:model" 格式搭配各主要 LLM provider。
//
// 了解更多：https://js.langchain.com/docs/integrations/chat/

/**
 * 此 workshop 中所有 agents 使用的 default model
 */
export const defaultModel = await initChatModel("openai:o3-mini");

/**
 * 若要使用不同 provider，請以下列範例之一取代下面那行：
 *
 * Azure OpenAI:
 * export const defaultModel = await initChatModel("azure_openai:gpt-4o", {
 *   azureOpenAIApiKey: process.env.AZURE_OPENAI_API_KEY,
 *   azureOpenAIApiInstanceName: "your-instance-name",
 *   azureOpenAIApiDeploymentName: "your-deployment-name",
 *   azureOpenAIApiVersion: process.env.AZURE_OPENAI_API_VERSION,
 * });
 *
 * Anthropic Claude:
 * export const defaultModel = await initChatModel("anthropic:claude-3-5-sonnet-20241022", {
 *   apiKey: process.env.ANTHROPIC_API_KEY,
 * });
 *
 * AWS Bedrock (Claude):
 * export const defaultModel = await initChatModel("bedrock:anthropic.claude-3-5-sonnet-20241022-v2:0", {
 *   region: process.env.AWS_REGION || "us-east-1",
 *   credentials: {
 *     accessKeyId: process.env.AWS_ACCESS_KEY_ID,
 *     secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
 *   },
 * });
 */

/**
 * 用於初始化特定 model 的 helper
 * 當某個特定 agent 想使用不同於 default 的 model 時使用
 * 
 * 範例：const gpt4 = await getModel("openai:gpt-4o");
 */
export async function getModel(modelName: string = "openai:03-mini") {
  return await initChatModel(modelName);
}

// ============================================================================
// Shared State 定義
// ============================================================================
//
// LANGGRAPH 中的 STATE
// State 是 nodes 在 LangGraph 中溝通的方式。它像是流經 graph 的 shared context，
// 每個 node 都會從中讀取並寫入。
//
// 為什麼使用 ZOD？
// 我們使用 Zod schemas 定義 state，以提供型別安全與驗證。
// withLangGraph() wrapper 會加入 message handling 所需的特殊 metadata。
//
// 了解更多：https://langchain-ai.github.io/langgraphjs/concepts/low_level/#state

/**
 * 所有 agents 共用的 state schema
 * 這能確保 message passing 與 context sharing 時的 state structure 一致
 * 
 * 欄位：
 * - messages：Conversation history（必填）
 * - customerId：已驗證的 customer identifier（選填）
 * - loadedMemory：從 memory store 載入的 user preferences
 * - remainingSteps：timeout 前允許的最大 steps（避免 infinite loops）
 */
export const AgentState = z.object({
  messages: withLangGraph(z.custom<BaseMessage[]>(), MessagesZodMeta),
  customerId: z.number().optional(),
  loadedMemory: z.string().default(""),
  remainingSteps: z.number().default(25),
});

export type AgentState = z.infer<typeof AgentState>;

// ============================================================================
// Database Setup
// ============================================================================
//
// WORKSHOP DATABASE
// 此 workshop 使用 Chinook database，這是一個範例 music store database。
// 它包含 artists、albums、tracks、invoices、customers 等 tables。
//
// 為什麼使用 SQL.JS？
// sql.js 會完全在記憶體中執行 SQLite，不需要 database server！
// 很適合 demos 和 workshops。
//
// PRODUCTION NOTE：
// 在 production 中，你會使用適當的 TypeORM configuration
// 連接真正的 database，例如 PostgreSQL、MySQL 等。

/**
 * 使用 sql.js 設定並初始化 Chinook database
 * 
 * Chinook database 是範例 music store database，包含：
 * - Artists, Albums, Tracks
 * - Customers, Invoices
 * - Employees, Playlists, Genres
 * 
 * @returns Promise<SqlDatabase> - 已初始化的 SqlDatabase instance
 */
export async function setupDatabase(): Promise<SqlDatabase> {
  console.log("📦 Setting up Chinook database...");

  // 從 GitHub 下載 Chinook SQL script
  const sqlScriptUrl =
    "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql";

  const response = await fetch(sqlScriptUrl);
  if (!response.ok) {
    throw new Error(
      `Failed to download SQL script. Status: ${response.status}`
    );
  }
  const sqlScript = await response.text();

  // 初始化 sql.js（編譯為 WebAssembly 的 SQLite）
  const SQL = await initSqlJs();
  const sqlJsDb = new SQL.Database();

  // 執行 SQL script 以建立並填入所有 tables
  sqlJsDb.exec(sqlScript);

  // 將 database 匯出為 buffer，讓 TypeORM 可以使用
  const dbBuffer = sqlJsDb.export();

  // 建立 TypeORM DataSource
  // TypeORM 對 raw SQL 提供良好的抽象
  const datasource = new DataSource({
    type: "sqljs",
    database: dbBuffer,
    synchronize: false,  // 不要自動 migrate schema
  });

  // 初始化 DataSource
  await datasource.initialize();

  // 包裝成 LangChain 的 SqlDatabase，供 agent 使用
  const db = await SqlDatabase.fromDataSourceParams({
    appDataSource: datasource,
  });

  console.log("✅ Chinook database loaded successfully!");
  return db;
}
