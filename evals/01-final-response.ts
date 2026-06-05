import "dotenv/config";
import { Client } from "langsmith";
import { evaluate } from "langsmith/evaluation";
import { v4 as uuidv4 } from "uuid";
import { Command } from "@langchain/langgraph";
import { createLLMAsJudge } from "openevals";
import { CORRECTNESS_PROMPT } from "openevals/prompts";
import { initChatModel } from "langchain";
import { z } from "zod/v3";
import { graph as multiAgentVerifyGraph } from "../agents/04-supervisor_with_verification.js";
import { getOrCreateDataset, logEvaluationSummary } from "./utils.js";

// ============================================================================
// 資料集定義
// ============================================================================

const DATASET_NAME = "LangGraph-Teaching-Tools 多代理：最終回覆（TypeScript）";

const examples = [
  {
    inputs: {
      messages: [
        {
          role: "user",
          content:
            "我的名字是 Aaron Mitchell，帳號 ID 是 32，帳號電話是 +1 (204) 452-6452。請幫我找最近一次歌曲購買的發票號碼。",
        },
      ],
    },
    outputs: {
      messages: [
        {
          role: "ai",
          content: "最近一次購買的發票 ID 是 342。",
        },
      ],
    },
  },
  {
    inputs: {
      messages: [{ role: "user", content: "我想申請退款。" }],
    },
    outputs: {
      messages: [
        {
          role: "ai",
          content: "我已確認你的帳號。請提供你想退款的購買項目細節。",
        },
      ],
    },
  },
  {
    inputs: {
      messages: [
        { role: "user", content: "Wish You Were Here 是誰錄製的？" },
      ],
    },
    outputs: {
      messages: [
        { role: "ai", content: "Wish You Were Here 是 Pink Floyd 的專輯。" },
      ],
    },
  },
  {
    inputs: {
      messages: [
        { role: "user", content: "你們有 Coldplay 的哪些專輯？" },
      ],
    },
    outputs: {
      messages: [
        {
          role: "ai",
          content: "我查詢了音樂商店資料庫，目前目錄中沒有 Coldplay 的專輯。",
        },
      ],
    },
  },
  {
    inputs: {
      messages: [{ role: "user", content: "我要怎麼成為億萬富翁？" }],
    },
    outputs: {
      messages: [
        {
          role: "ai",
          content:
            "我能協助處理數位音樂商店相關問題。如果你有音樂目錄或過往購買紀錄的問題，可以再告訴我。",
        },
      ],
    },
  },
];

// ============================================================================
// 應用邏輯
// ============================================================================

/**
 * 執行 multi-agent graph，並處理 human-in-the-loop 的中斷與恢復。
 */
async function runGraph(
  inputs: Record<string, any>
): Promise<Record<string, any>> {
  const threadId = uuidv4();
  const config = { configurable: { thread_id: threadId, user_id: "10" } };

  await multiAgentVerifyGraph.invoke(inputs, config);

  const result: any = await multiAgentVerifyGraph.invoke(
    new Command({ resume: "我的客戶 ID 是 10" }),
    { configurable: { thread_id: threadId, user_id: "10" } }
  );

  const content = String(result.messages[result.messages.length - 1].content);
  return { messages: [{ role: "ai", content }] };
}

// ============================================================================
// 評估器
// ============================================================================

const evalModel = await initChatModel("openai:gpt-4o-mini");

const baseCorrectness = createLLMAsJudge({
  prompt: CORRECTNESS_PROMPT,
  feedbackKey: "correctness",
  judge: evalModel,
});

async function correctnessEvaluator(args: any) {
  return baseCorrectness({
    inputs: args.inputs,
    outputs: args.outputs,
    referenceOutputs: args.referenceOutputs,
  });
}

const professionalismGraderInstructions = `你是一位評估器，負責評估代理回覆是否具備專業度。
你會收到一個問題、代理回覆，以及標準參考回覆。
請依照以下專業度準則進行評估：

(1) 語氣：回覆全程應維持尊重、禮貌，並符合商務情境。
(2) 語言：回覆應使用正確文法、拼字與專業用語，避免俚語、過度隨意或不適當的表達。
(3) 結構：回覆應組織良好、清楚且容易理解。
(4) 禮貌：回覆應適當回應使用者請求，並尊重使用者的時間與關切。
(5) 界線：回覆應維持適當的專業界線，避免過度熟絡或非正式。
(6) 幫助性：回覆應展現真誠協助使用者的意圖，且符合專業標準。

專業度評分：
True 代表代理回覆在所有準則上都符合專業標準。
False 代表代理回覆在一個或多個重要面向上未達專業標準。

請逐步說明你的推理，確保評估完整且公正。`;

const ProfessionalismGradeSchema = z.object({
  reasoning: z
    .string()
    .describe(
      "逐步說明你對專業度評估的推理，需涵蓋語氣、語言、結構、禮貌、界線與幫助性。"
    ),
  isProfessional: z
    .boolean()
    .describe("如果代理回覆符合專業標準則為 True，否則為 False。"),
});

const professionalismGraderLlm = evalModel.withStructuredOutput(
  ProfessionalismGradeSchema
);

async function professionalismEvaluator({
  outputs,
  referenceOutputs,
  inputs,
}: {
  outputs: Record<string, any>;
  referenceOutputs?: Record<string, any>;
  inputs?: Record<string, any>;
}): Promise<{ key: string; score: number; comment: string }> {
  const userContext = `問題：${JSON.stringify(inputs?.messages)}
標準參考回覆：${JSON.stringify(referenceOutputs?.messages)}
代理回覆：${JSON.stringify(outputs?.messages)}`;

  const grade: any = await professionalismGraderLlm.invoke([
    { role: "system", content: professionalismGraderInstructions },
    { role: "user", content: userContext },
  ]);

  return {
    key: "professionalism",
    score: grade.isProfessional ? 1 : 0,
    comment: grade.reasoning,
  };
}

// ============================================================================
// 主要評估流程
// ============================================================================

async function main() {
  console.log("開始 Final Response 評估\n");

  const client = new Client();
  await getOrCreateDataset(client, DATASET_NAME, examples);

  console.log("\n正在執行評估，可能需要幾分鐘...\n");

  await evaluate((inputs: any) => runGraph(inputs), {
    data: DATASET_NAME,
    evaluators: [correctnessEvaluator, professionalismEvaluator],
    experimentPrefix: "agent-e2e",
    maxConcurrency: 3,
    client,
  });

  logEvaluationSummary("Final Response (E2E)", DATASET_NAME);

  console.log("請至 LangSmith 查看詳細結果與視覺化資料");
}

// 執行 evaluation。
main().catch(console.error);
