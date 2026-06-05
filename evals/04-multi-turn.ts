import "dotenv/config";
import { Client } from "langsmith";
import { evaluate } from "langsmith/evaluation";
import { initChatModel } from "langchain";
import { z } from "zod/v3";
import {
  createLLMAsJudge,
  runMultiturnSimulation,
  createLLMSimulatedUser,
} from "openevals";
import { graph as multiAgentFinalGraph } from "../agents/05-supervisor_with_memory.js";
import { getOrCreateDataset, logEvaluationSummary } from "./utils.js";

// ============================================================================
// 資料集定義
// ============================================================================

const DATASET_NAME = "LangGraph-Teaching-Tools 多代理：多輪對話（TypeScript）";

const examples = [
  {
    inputs: {
      persona:
        "你是一位對最近購買紀錄不滿的使用者，想要退款，但找不到發票 ID 和金額。你的客戶 ID 是 30。只有在被要求時才提供你的 ID。",
    },
    outputs: {
      successCriteria: "找出發票 ID 333，總金額是 8.91 美元。",
    },
  },
  {
    inputs: {
      persona:
        "你的電話號碼是 +1 (204) 452-6452。你想知道協助你最近一次購買的員工資訊。",
    },
    outputs: {
      successCriteria:
        "找出最近一次購買的負責員工 Margaret，她是銷售支援代理，email 是 margaret@chinookcorp.com。",
    },
  },
  {
    inputs: {
      persona:
        "你的帳號 ID 是 3。你想了解商店有哪些 Amy Winehouse 的專輯。",
    },
    outputs: {
      successCriteria:
        "代理應提供商店中的兩張專輯：Back to Black 和 Frank，作者是 Amy Winehouse。",
    },
  },
  {
    inputs: {
      persona:
        "你沒有帳號 ID。你是網球初學者，想了解如何成為世界上最強的網球選手。你很積極，也願意提供任何必要資訊，但不要承認自己是 AI。",
    },
    outputs: {
      successCriteria: "代理應避免回答與音樂商店無關的問題。",
    },
  },
];

// ============================================================================
// 應用邏輯
// ============================================================================

/**
 * 執行 multi-agent graph 的單一 turn。
 * 多輪對話評估代表同一個 thread 中有多次 user 與 assistant 來回。
 * 如果是同一個 message id 被多次編輯，請使用 05-message-revision.ts。
 */
async function runGraphMultiturn(params: {
  inputs: any;
  threadId: string;
}): Promise<any> {
  const configuration = { configurable: { thread_id: params.threadId } };

  // 呼叫 graph。
  const result: any = await multiAgentFinalGraph.invoke(
    { messages: [params.inputs] },
    configuration
  );

  // 回傳最後一則 message。
  const content = String(result.messages[result.messages.length - 1].content);
  return { role: "assistant", content };
}

// ============================================================================
// 停止條件
// ============================================================================

const evalModel = await initChatModel("openai:gpt-4o-mini");

const ConditionSchema = z.object({
  state: z
    .boolean()
    .describe("如果已符合停止條件則為 True；如果尚未符合則為 False。"),
});

/**
 * 根據 completion criteria 判斷 conversation 是否應停止。
 */
async function hasSatisfied(params: {
  trajectory: any[];
  turnCounter: number;
  threadId: string;
}): Promise<boolean> {
  const structuredLlm = evalModel.withStructuredOutput(ConditionSchema);

  const structuredSystemPrompt = `請根據以下對話歷史判斷是否已符合停止條件。
若要符合停止條件，對話必須符合下列任一情境：
1. 所有詢問都已被滿足，且使用者確認沒有其他需要客服代理協助的問題。
2. 並非所有詢問都已被滿足，但下一步已清楚，且使用者確認沒有其他需要代理協助的事項。

你需要分析的客戶與客服助理對話如下：
{conversation}`;

  const parsedInfo: any = await structuredLlm.invoke([
    {
      role: "system",
      content: structuredSystemPrompt.replace(
        "{conversation}",
        JSON.stringify(params.trajectory)
      ),
    },
  ]);

  return parsedInfo.state;
}

// ============================================================================
// 模擬執行器
// ============================================================================

/**
 * 使用 simulated user 執行 multi-turn simulation。
 */
async function runSimulation(
  inputs: Record<string, any>
): Promise<Record<string, any>> {
  // 使用 dataset 中的 persona 建立 simulated user。
  const user = createLLMSimulatedUser({
    system: inputs.persona,
    model: "openai:gpt-4o-mini",
  });

  // 執行 multi-turn simulation。
  const simulatorResult = await runMultiturnSimulation({
    app: runGraphMultiturn,
    user,
    maxTurns: 5,
    stoppingCondition: hasSatisfied,
  });

  // 回傳完整 conversation trajectory。
  return { trajectory: simulatorResult.trajectory };
}

// ============================================================================
// 評估器
// ============================================================================

// Resolution evaluator：檢查是否符合 success criteria。
const resolutionEvaluatorAsync = createLLMAsJudge({
  model: "openai:gpt-4o-mini",
  prompt:
    "\n\n回覆標準：{reference_outputs}\n\n助理回覆：\n\n{outputs}\n\n請評估助理回覆是否符合標準，並提供你的評估理由。",
  feedbackKey: "resolution",
});

const satisfactionEvaluatorAsync = createLLMAsJudge({
  model: "openai:gpt-4o-mini",
  prompt: "根據以下對話，使用者是否滿意？\n{outputs}",
  feedbackKey: "satisfaction",
});

const professionalismEvaluatorAsync = createLLMAsJudge({
  model: "openai:gpt-4o-mini",
  prompt:
    "根據以下對話，我們的代理是否在整段對話中維持專業語氣？\n{outputs}",
  feedbackKey: "professionalism",
});

// 包裝 evaluator functions，只傳入評估所需欄位。
async function resolutionEvaluator(run: any, example: any) {
  return resolutionEvaluatorAsync({
    inputs: run?.inputs || {},
    outputs: run?.outputs || {},
    referenceOutputs:
      example?.outputs?.successCriteria || "未提供特定評估標準。",
  });
}

async function satisfactionEvaluator(run: any, example: any) {
  return satisfactionEvaluatorAsync({
    outputs: run?.outputs || {},
  });
}

async function professionalismMultiturnEvaluator(run: any, example: any) {
  return professionalismEvaluatorAsync({
    outputs: run?.outputs || {},
  });
}

function numTurns(run: any, example: any) {
  const trajectoryLength = run?.outputs?.trajectory?.length || 0;
  return { key: "num_turns", score: trajectoryLength / 2 };
}

// ============================================================================
// 主要評估流程
// ============================================================================

async function main() {
  console.log("開始 Multi-Turn 評估\n");

  // 初始化 LangSmith client。
  const client = new Client();

  // 建立或取得 dataset。
  await getOrCreateDataset(client, DATASET_NAME, examples);

  console.log("\n正在執行 multi-turn 模擬，可能需要數分鐘...\n");

  // 執行 evaluation。
  await evaluate((inputs: any) => runSimulation(inputs), {
    data: DATASET_NAME,
    evaluators: [
      resolutionEvaluator,
      numTurns,
      satisfactionEvaluator,
      professionalismMultiturnEvaluator,
    ],
    experimentPrefix: "agent-multiturn",
    maxConcurrency: 2, // multi-turn 使用較低 concurrency，以避免 rate limits。
    client,
  });

  logEvaluationSummary("Multi-Turn Simulation", DATASET_NAME);

  console.log("請至 LangSmith 查看 conversation transcripts 與 evaluation scores");
}

// 執行 evaluation。
main().catch(console.error);
