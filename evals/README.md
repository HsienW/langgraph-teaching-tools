# 代理評估

這個目錄包含可直接執行的 TypeScript 評估腳本，用來測試代理在不同場景下的表現。

## 評估如何運作

每個評估通常包含三個部分：

1. **資料集**：範例輸入與預期輸出
2. **代理程式**：被測試的 graph 或 agent
3. **評估器**：計算分數的函式，例如正確性、路由、工具呼叫、專業度

## 評估類型

### 1. 最終回覆評估 (`01-final-response.ts`)

將代理視為黑盒，評估最終回答是否完成任務。

- **輸入**：`messages`
- **輸出**：代理最終回覆
- **評估器**：正確性、專業度

```bash
npx tsx evals/01-final-response.ts
```

### 2. 單一步驟評估 (`02-single-step.ts`)

評估 supervisor 是否把使用者問題路由到正確 subagent。

- **輸入**：`messages`
- **輸出**：被選中的 route
- **評估器**：路由正確性

```bash
npx tsx evals/02-single-step.ts
```

### 3. 軌跡評估 (`03-trajectory.ts`)

評估代理執行過程中的 tool calls 是否符合預期。

- **輸入**：`question`
- **輸出**：tool call trajectory
- **評估器**：完全符合、未匹配步驟、必要工具、禁止工具、必要順序、呼叫次數上限

```bash
npx tsx evals/03-trajectory.ts
```

#### 外部 tool 呼叫次數不固定時

如果外部 tool 需要分頁、重試、輪詢或 while loop 才能拿到結果，完整軌跡不會固定。這時不要只用 exact match，應該改用彈性評估：

- **requiredTools**：必要 tool 必須出現
- **forbiddenTools**：禁止 tool 不可出現
- **orderedRequiredTools**：必要 tool 需依順序出現，但中間允許額外呼叫
- **maxToolCalls**：限制呼叫次數，避免 retry 或 loop 失控

範例：

```typescript
{
  inputs: {
    question: "我的客戶 ID 是 10。請查詢最近一次購買的發票資訊；如果外部服務需要分頁、重試或輪詢，請持續查到可用結果為止。",
  },
  outputs: {
    requiredTools: [
      "invoice_information_subagent",
      "get_invoices_by_customer_sorted_by_date",
    ],
    orderedRequiredTools: [
      "invoice_information_subagent",
      "get_invoices_by_customer_sorted_by_date",
    ],
    forbiddenTools: ["music_catalog_subagent"],
    maxToolCalls: 12,
  },
}
```

### 4. 多輪對話評估 (`04-multi-turn.ts`)

模擬同一個 thread 中使用者與助理多次來回，評估完整互動是否達成目標。

- **輸入**：使用者 persona
- **輸出**：完整 conversation trajectory
- **評估器**：解決度、滿意度、專業度、輪次數

```bash
npx tsx evals/04-multi-turn.ts
```

#### 什麼算多輪對話？

多輪對話代表同一個 thread 或 chat id 下有多次 user / assistant 來回：

```text
使用者第 1 輪
助理第 1 輪
使用者第 2 輪
助理第 2 輪
```

如果使用者只是同一個 message id 多次編輯，這不是典型 multi-turn evaluation，請使用 `05-message-revision.ts`。

### 5. 同訊息編輯評估 (`05-message-revision.ts`)

評估同一個 message id 被多次編輯時，代理是否只根據最新版處理需求，並避免被舊版本污染。

- **輸入**：同一個 `messageId` 的多個 `revisions`
- **輸出**：最新版 message 與代理回覆
- **評估器**：最新版需求覆蓋、舊版內容避開、最新版輸入使用確認

```bash
npx tsx evals/05-message-revision.ts
```

#### 同 message id 多次編輯時

這種情境比較像 revision evaluation，而不是 multi-turn evaluation。常見策略是只把最後一版 message 送進代理，然後檢查：

- **最新版需求是否被滿足**
- **舊版已被覆寫的意圖是否沒有污染回覆**
- **執行時是否真的使用最後一版 message**

範例：

```typescript
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
        content: "請不要處理退款。我的客戶 ID 是 10，請改成查詢最近一次購買的發票號碼。",
      },
    ],
  },
  outputs: {
    expectedLatestIntent: "查詢最近一次購買的發票號碼",
    requiredKeywords: ["發票"],
    forbiddenKeywords: ["退款", "退費"],
    shouldUseLatestRevisionOnly: true,
  },
}
```

## 前置條件

請確認 `.env` 已設定 LangSmith API key：

```bash
LANGSMITH_API_KEY=your-api-key-here
```

安裝依賴：

```bash
pnpm install
```

## 執行所有評估

```bash
npx tsx evals/01-final-response.ts
npx tsx evals/02-single-step.ts
npx tsx evals/03-trajectory.ts
npx tsx evals/04-multi-turn.ts
npx tsx evals/05-message-revision.ts
```

## 工具函式

`utils.ts` 包含共用 helper functions：

- `getOrCreateDataset()`：只在 dataset 不存在時建立
- `extractToolCalls()`：從代理執行中解析工具呼叫
- `logEvaluationSummary()`：格式化 console 輸出

## 注意事項

- **Datasets 會持久保存**：建立後會永久儲存測試案例
- **每次執行會建立新的 experiment**：可用來比較不同版本的代理
- **需要 LangSmith**：執行評估需要 LangSmith 帳號與 API key
- **需要可用代理**：評估會呼叫 `../agents/` 中的代理，請先確認代理本身可執行