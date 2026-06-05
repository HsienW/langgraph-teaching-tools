import "dotenv/config";
import { Client } from "langsmith";
import { evaluate } from "langsmith/evaluation";
import { v4 as uuidv4 } from "uuid";
import { graph as multiAgentFinalGraph } from "../agents/05-supervisor_with_memory.js";
import { getOrCreateDataset, logEvaluationSummary } from "./utils.js";

// ============================================================================
// 資料集定義
// ============================================================================

const DATASET_NAME = "LangGraph-Teaching-Tools 多代理：同訊息編輯評估（TypeScript）";

const examples = [
  {
    inputs: {
      messageId: "msg_001",
      revisions: [
        {
          version: 1,
          content: "我想申請退款。",
        },
        {
          version: 2,
          content:
            "請不要處理退款。我的客戶 ID 是 10，請改成查詢最近一次購買的發票號碼。",
        },
      ],
    },
    outputs: {
      expectedLatestIntent: "查詢最近一次購買的發票號碼",
      requiredKeywords: ["發票"],
      forbiddenKeywords: ["退款", "退費"],
      shouldUseLatestRevisionOnly: true,
    },
  },
  {
    inputs: {
      messageId: "msg_002",
      revisions: [
        {
          version: 1,
          content: "我的帳號 ID 是 3。請查詢 Amy Winehouse 的專輯。",
        },
        {
          version: 2,
          content:
            "我的帳號 ID 是 3。請改成查詢 Amy Winehouse 的歌曲，不要查專輯。",
        },
      ],
    },
    outputs: {
      expectedLatestIntent: "查詢 Amy Winehouse 的歌曲",
      requiredKeywords: ["Amy Winehouse"],
      forbiddenKeywords: ["專輯"],
      shouldUseLatestRevisionOnly: true,
    },
  },
  {
    inputs: {
      messageId: "msg_003",
      revisions: [
        {
          version: 1,
          content: "我的客戶 ID 是 1。請查詢最近一次購買紀錄。",
        },
        {
          version: 2,
          content:
            "我的客戶 ID 是 1。請查詢最近一次購買紀錄，並補充該筆購買的總金額。",
        },
      ],
    },
    outputs: {
      expectedLatestIntent: "查詢最近一次購買紀錄與總金額",
      requiredKeywords: ["總金額"],
      forbiddenKeywords: [],
      shouldUseLatestRevisionOnly: true,
    },
  },
];

// ============================================================================
// 應用邏輯
// ============================================================================

function getLatestRevision(inputs: Record<string, any>): string {
  const revisions = inputs.revisions || [];
  const latestRevision = revisions[revisions.length - 1];
  return latestRevision?.content || "";
}

/**
 * 同一個 message id 多次編輯不是典型 multi-turn conversation。
 * 這裡只把最新版送進 graph，評估代理是否避免被已覆寫的舊版本污染。
 */
async function runLatestRevision(
  inputs: Record<string, any>
): Promise<Record<string, any>> {
  const latestMessage = getLatestRevision(inputs);
  const threadId = uuidv4();
  const configuration = { configurable: { thread_id: threadId, user_id: "10" } };

  const result: any = await multiAgentFinalGraph.invoke(
    { messages: [{ role: "user", content: latestMessage }] },
    configuration
  );

  const content = String(result.messages[result.messages.length - 1].content);

  return {
    messageId: inputs.messageId,
    latestMessage,
    finalResponse: content,
  };
}

// ============================================================================
// 評估器
// ============================================================================

function includesAnyKeyword(text: string, keywords: string[]): boolean {
  return keywords.some((keyword) => text.includes(keyword));
}

function includesAllKeywords(text: string, keywords: string[]): boolean {
  return keywords.every((keyword) => text.includes(keyword));
}

/**
 * 評估回覆是否包含最新版需求中的必要資訊。
 */
function latestIntentEvaluator({
  outputs,
  referenceOutputs,
}: {
  outputs: Record<string, any>;
  referenceOutputs?: Record<string, any>;
}): { key: string; score: number } {
  const finalResponse = String(outputs?.finalResponse || "");
  const requiredKeywords = referenceOutputs?.requiredKeywords || [];

  return {
    key: "latest_intent_covered",
    score: includesAllKeywords(finalResponse, requiredKeywords) ? 1 : 0,
  };
}

/**
 * 評估回覆是否沒有使用被舊版本覆寫的內容。
 */
function supersededContentEvaluator({
  outputs,
  referenceOutputs,
}: {
  outputs: Record<string, any>;
  referenceOutputs?: Record<string, any>;
}): { key: string; score: number } {
  const finalResponse = String(outputs?.finalResponse || "");
  const forbiddenKeywords = referenceOutputs?.forbiddenKeywords || [];

  return {
    key: "superseded_content_avoided",
    score: includesAnyKeyword(finalResponse, forbiddenKeywords) ? 0 : 1,
  };
}

/**
 * 評估執行時是否確實使用最後一版 message。
 */
function latestRevisionInputEvaluator({
  inputs,
  outputs,
}: {
  inputs: Record<string, any>;
  outputs: Record<string, any>;
}): { key: string; score: number } {
  const expectedLatestMessage = getLatestRevision(inputs);

  return {
    key: "latest_revision_used",
    score: outputs?.latestMessage === expectedLatestMessage ? 1 : 0,
  };
}

// ============================================================================
// 主要評估流程
// ============================================================================

async function main() {
  console.log("開始 Message Revision 評估\n");

  const client = new Client();
  await getOrCreateDataset(client, DATASET_NAME, examples);

  console.log("\n正在執行同訊息編輯評估...\n");

  await evaluate((inputs: any) => runLatestRevision(inputs), {
    data: DATASET_NAME,
    evaluators: [
      latestIntentEvaluator,
      supersededContentEvaluator,
      latestRevisionInputEvaluator,
    ],
    experimentPrefix: "agent-message-revision",
    maxConcurrency: 3,
    client,
  });

  logEvaluationSummary("Message Revision", DATASET_NAME);

  console.log("請至 LangSmith 查看同訊息編輯評估結果");
}

// 執行 evaluation。
main().catch(console.error);
