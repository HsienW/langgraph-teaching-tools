import "dotenv/config";
import { Client } from "langsmith";
import { evaluate } from "langsmith/evaluation";
import { v4 as uuidv4 } from "uuid";
import { HumanMessage } from "langchain";
import { graph as supervisor } from "../agents/03-supervisor.js";
import { getOrCreateDataset, logEvaluationSummary } from "./utils.js";

// ============================================================================
// 資料集定義
// ============================================================================

const DATASET_NAME = "LangGraph-Teaching-Tools 多代理：單一步驟（TypeScript）";

const examples = [
  {
    inputs: {
      messages: "我的客戶 ID 是 1。請查詢我最近一次購買紀錄。",
    },
    outputs: { route: "invoice_information_subagent" },
  },
  {
    inputs: { messages: "請查詢 U2 有哪些歌曲。" },
    outputs: { route: "music_catalog_subagent" },
  },
  {
    inputs: {
      messages:
        "我的名字是 Aaron Mitchell，帳號電話是 +1 (204) 452-6452。請幫我找最近一次歌曲購買的發票號碼。",
    },
    outputs: { route: "invoice_information_subagent" },
  },
  {
    inputs: {
      messages:
        "Wish You Were Here 是誰錄製的？你們還有他們的哪些專輯？",
    },
    outputs: { route: "music_catalog_subagent" },
  },
  {
    inputs: { messages: "今年溫布頓網球錦標賽是誰贏了？" },
    outputs: { route: "model" }, // 最後一則 message 應來自 supervisor；不呼叫任何 sub-agents。
  },
];

// ============================================================================
// 應用邏輯
// ============================================================================

/**
 * 執行 supervisor，並擷取它選擇的 route。
 */
async function runSupervisorRouting(
  inputs: Record<string, any>
): Promise<Record<string, any>> {
  const result: any = await supervisor.invoke(
    {
      messages: [new HumanMessage(inputs.messages)],
      customerId: 10,
      loadedMemory: "",
      remainingSteps: 25,
    },
    {
      interruptAfter: ["tools"],
      configurable: { thread_id: uuidv4(), user_id: "10" },
    }
  );

  const lastMessage = result.messages[result.messages.length - 1];
  const route = lastMessage.name || "supervisor";

  return { route };
}

// ============================================================================
// 評估器
// ============================================================================

/**
 * 評估 agent 是否選擇正確 route。
 */
function correctRouteEvaluator({
  outputs,
  referenceOutputs,
}: {
  outputs: Record<string, any>;
  referenceOutputs?: Record<string, any>;
}): { key: string; score: number } {
  const isCorrect = outputs?.route === referenceOutputs?.route;
  return {
    key: "correct",
    score: isCorrect ? 1 : 0,
  };
}

// ============================================================================
// 主要評估流程
// ============================================================================

async function main() {
  console.log("開始 Single-Step 評估\n");

  const client = new Client();
  await getOrCreateDataset(client, DATASET_NAME, examples);

  console.log("\n正在執行評估...\n");

  await evaluate((inputs: any) => runSupervisorRouting(inputs), {
    data: DATASET_NAME,
    evaluators: [correctRouteEvaluator],
    experimentPrefix: "agent-singlestep",
    maxConcurrency: 3,
    client,
  });

  logEvaluationSummary("Single-Step (Routing)", DATASET_NAME);

  console.log("請至 LangSmith 查看詳細結果");
}

// 執行 evaluation。
main().catch(console.error);
