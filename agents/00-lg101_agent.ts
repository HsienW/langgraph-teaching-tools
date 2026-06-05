import "dotenv/config";
import { createAgent, tool } from "langchain";
import { z } from "zod/v3";
import { defaultModel } from "./utils";

// ============================================================================
// Tool 定義
// ============================================================================
// 
// 什麼是 tools？
// Tools 是 LLM 可以呼叫的函數，用來執行動作或擷取資訊。
// 它們是建置可與外部系統互動的 agents 時不可或缺的部分。
// 
// 可以把 tools 想成是賦予 LLM「超能力」：它不只能產生文字，
// 現在也能查天氣、查詢資料庫、呼叫 API 等。

type WeatherApiResponse = {
  current: {
    temperature_2m: number;
    weather_code: number;
  };
};

// tool() 函數會從一般的 TypeScript 函數建立 LangChain tool。
// 它接受兩個參數：
// 1. 實作函數，也就是 tool 實際執行的內容
// 2. 包含 name、description 和 schema 的設定物件
const getWeather = tool(
  // 實作：當 LLM 決定使用此 tool 時，會呼叫這個 async 函數
  async ({ latitude, longitude }) => {
    const url = new URL("https://api.open-meteo.com/v1/forecast");
    url.searchParams.set("latitude", latitude.toString());
    url.searchParams.set("longitude", longitude.toString());
    url.searchParams.set("current", "temperature_2m,weather_code");
    url.searchParams.set("temperature_unit", "fahrenheit");

    const response = await fetch(url);
    const data: WeatherApiResponse = await response.json();
    
    // Tools 一律回傳字串，LLM 會處理結果
    return JSON.stringify({
      temperature_fahrenheit: data.current.temperature_2m,
      weather_code: data.current.weather_code,
    });
  },
  {
    // Tool name：LLM 用它來識別要呼叫哪個 tool
    name: "get_weather",
    
    // Description：這非常重要！LLM 會用它來決定何時使用此 tool。
    // 請寫清楚且詳細的描述，說明 tool 的作用與使用時機。
    description:
      "Get current temperature in Fahrenheit and weather code for given coordinates. Returns JSON with temperature_fahrenheit and weather_code (do not include the code in your response, translate it to plain English)",
    
    // Schema：使用 Zod 定義 tool 的參數，以提供型別安全與驗證。
    // LLM 會使用這裡的 descriptions 來理解應提供哪些值。
    schema: z.object({
      latitude: z.number().describe("Latitude coordinate"),
      longitude: z.number().describe("Longitude coordinate"),
    }),
  }
);

// ============================================================================
// Agent 建立
// ============================================================================
//
// 簡單做法：createAgent()
// 
// createAgent() 函數是在 LangChain 中建立 agent 最簡單的方式。
// 它會自動設定一個 ReAct 風格的 agent，能夠推理並使用 tools。
// 
// 在底層，它會建立一個 LangGraph，裡面包含 LLM 和 tools 的 nodes，
// 但剛開始使用時不需要擔心這些細節。

const agent = createAgent({
  // model：驅動此 agent 的 LLM
  model: defaultModel,
  
  // tools：agent 可使用的 tools 陣列
  // agent 會自動決定何時以及如何使用這些 tools
  tools: [getWeather],
  
  // systemPrompt：引導 agent 行為的指令
  // 這會設定 agent 的個性、角色與準則
  systemPrompt:
    "您是一位得力的天氣助手。請使用 get_weather 工具查詢各城市的天氣狀況。",
});

// ============================================================================
// 匯出
// ============================================================================
//
// 我們匯出 agent.graph，也就是已編譯、可被呼叫的 LangGraph。
// 這個 graph 可以在本機執行，也可以部署到 LangSmith Deployments。
//
// 使用範例：
//   const result = await graph.invoke({ messages: [{ role: "user", content: "What's the weather in SF?" }] });

export const graph = agent.graph;
