import { Client } from "langsmith";

// ============================================================================
// 型別
// ============================================================================

export interface EvaluationResult {
  key: string;
  score: number | boolean;
  comment?: string;
}

export interface DatasetExample {
  inputs: Record<string, any>;
  outputs: Record<string, any>;
}

// ============================================================================
// 資料集輔助函數
// ============================================================================

/**
 * 如果 LangSmith dataset 尚不存在，則建立它。
 * 使用 readDataset 檢查是否存在，這比 hasDataset 更可靠。
 */
export async function getOrCreateDataset(
  client: Client,
  datasetName: string,
  examples: DatasetExample[]
): Promise<void> {
  try {
    // 嘗試讀取 dataset；如果存在就完成。
    await client.readDataset({ datasetName });
    console.log(`Dataset 已存在：${datasetName}`);
  } catch (error) {
    // Dataset 不存在，建立它。
    console.log(`正在建立 dataset：${datasetName}`);
    const dataset = await client.createDataset(datasetName);

    // 逐一建立 examples，符合目前的 LangSmith API。
    for (const example of examples) {
      await client.createExample({
        inputs: example.inputs,
        outputs: example.outputs,
        dataset_id: dataset.id,
      });
    }

    console.log(`已建立 dataset，包含 ${examples.length} 筆 examples`);
  }
}

// ============================================================================
// Tool Call 擷取
// ============================================================================

/**
 * 從 agent messages 擷取 tool call names。
 */
export function extractToolCalls(input: any): string[] {
  const toolCalls: string[] = [];

  if (input && typeof input === "object" && "messages" in input) {
    for (const message of input.messages) {
      if (message.additional_kwargs?.tool_calls) {
        const tools = message.additional_kwargs.tool_calls;
        toolCalls.push(...tools.map((tool: any) => tool.function.name));
      }
    }
  } else if (Array.isArray(input)) {
    for (const item of input) {
      if (item.name) {
        toolCalls.push(item.name);
      }
    }
  }

  return toolCalls;
}

// ============================================================================
// 記錄輔助函數
// ============================================================================

/**
 * 記錄 evaluation summary。
 */
export function logEvaluationSummary(experimentPrefix: string, datasetName: string) {
  console.log(`\n${"=".repeat(60)}`);
  console.log(`Evaluation 完成：${experimentPrefix}`);
  console.log(`Dataset：${datasetName}`);
  console.log(`${"=".repeat(60)}\n`);
}