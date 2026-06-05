import "dotenv/config";
import { z } from "zod/v3";
import { SystemMessage, AIMessage, tool } from "langchain";
import {
  StateGraph,
  START,
  END,
  MemorySaver,
  InMemoryStore,
} from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { SqlDatabase } from "@langchain/classic/sql_db";
import { setupDatabase, AgentState, defaultModel } from "./utils.js";

// ============================================================================
// 使用 LANGGRAPH PRIMITIVES 建置 AGENTS
// ============================================================================
//
// 此檔案示範如何使用 LangGraph primitives 建置 ReAct 風格的 agent。
// 不同於使用 createAgent() 的 00-lg101_agent.ts，這裡會手動建構
// 含有 nodes 與 edges 的 graph。這讓我們有更多控制權，也幫助你
// 理解底層實際發生的事情。
//
// 什麼是 REACT？
// ReAct = Reasoning + Acting。這是一種 agent 的運作模式：
// 1. 思考該做什麼（Reasoning）
// 2. 視需要使用 tool（Acting）
// 3. 觀察結果
// 4. 重複直到得到答案
//
// 為什麼要手動建置？
// - 對 agent 行為有更多控制權
// - 更了解 agents 的運作方式
// - 能自訂流程，例如加入 verification、memory 等
// - 此 agent 會成為較大型 supervisor workflow 的一部分

// ============================================================================
// Tools
// ============================================================================

async function createMusicTools(db: SqlDatabase) {
  const getAlbumsByArtist = tool(
    async ({ artist }) => {
      const query = `
        SELECT Album.Title, Artist.Name 
        FROM Album 
        JOIN Artist ON Album.ArtistId = Artist.ArtistId 
        WHERE Artist.Name LIKE '%${artist}%'
        LIMIT 8;
      `;
      const rawResult = await db.run(query);
      const result =
        typeof rawResult === "string" ? JSON.parse(rawResult) : rawResult;
      return JSON.stringify(result);
    },
    {
      name: "get_albums_by_artist",
      description: "Get albums by an artist.",
      schema: z.object({
        artist: z.string().describe("The artist name"),
      }),
    }
  );

  const getTracksByArtist = tool(
    async ({ artist }) => {
      const query = `
        SELECT Track.Name as SongName, Artist.Name as ArtistName 
        FROM Album 
        LEFT JOIN Artist ON Album.ArtistId = Artist.ArtistId 
        LEFT JOIN Track ON Track.AlbumId = Album.AlbumId 
        WHERE Artist.Name LIKE '%${artist}%'
        LIMIT 8;
      `;
      const rawResult = await db.run(query);
      const result =
        typeof rawResult === "string" ? JSON.parse(rawResult) : rawResult;
      return JSON.stringify(result);
    },
    {
      name: "get_tracks_by_artist",
      description: "Get songs by an artist (or similar artists).",
      schema: z.object({
        artist: z.string().describe("The artist name"),
      }),
    }
  );

  const getSongsByGenre = tool(
    async ({ genre }) => {
      // 先取得 genre ID
      const genreQuery = `SELECT GenreId FROM Genre WHERE Name LIKE '%${genre}%' LIMIT 8;`;
      const rawGenreResult = await db.run(genreQuery);
      const genreResult =
        typeof rawGenreResult === "string"
          ? JSON.parse(rawGenreResult)
          : rawGenreResult;

      if (!genreResult || genreResult.length === 0) {
        return `No songs found for the genre: ${genre}`;
      }

      const genreIds = genreResult.map((row: any) => row.GenreId).join(", ");

      const songsQuery = `
        SELECT Track.Name as SongName, Artist.Name as ArtistName
        FROM Track
        LEFT JOIN Album ON Track.AlbumId = Album.AlbumId
        LEFT JOIN Artist ON Album.ArtistId = Artist.ArtistId
        WHERE Track.GenreId IN (${genreIds})
        GROUP BY Artist.Name
        LIMIT 8;
      `;

      const rawSongs = await db.run(songsQuery);
      const songs =
        typeof rawSongs === "string" ? JSON.parse(rawSongs) : rawSongs;
      return JSON.stringify(songs);
    },
    {
      name: "get_songs_by_genre",
      description: "Fetch songs from the database that match a specific genre.",
      schema: z.object({
        genre: z.string().describe("The genre of the songs to fetch"),
      }),
    }
  );

  const checkForSongs = tool(
    async ({ songTitle }) => {
      const query = `SELECT * FROM Track WHERE Name LIKE '%${songTitle}%' LIMIT 8;`;
      const rawResult = await db.run(query);
      const result =
        typeof rawResult === "string" ? JSON.parse(rawResult) : rawResult;
      return JSON.stringify(result);
    },
    {
      name: "check_for_songs",
      description: "Check if a song exists by its name.",
      schema: z.object({
        songTitle: z.string().describe("The song title to search for"),
      }),
    }
  );

  return [getAlbumsByArtist, getTracksByArtist, getSongsByGenre, checkForSongs];
}

// ============================================================================
// System Prompt
// ============================================================================
//
// system prompt 會定義 agent 的角色、職責與行為。
// 這是給 music catalog assistant 使用的專門 prompt，
// 此 assistant 會成為較大型 multi-agent system 的一部分。

function generateMusicAssistantPrompt(memory: string = "None"): string {
  return `
<important_background>
You are a member of the assistant team, your role specifically is to focused on helping customers discover and learn about music in our digital catalog. 
If you are unable to find playlists, songs, or albums associated with an artist, it is okay. 
Just respond that the catalog does not have any playlists, songs, or albums associated with that artist.
You also have context on any saved user preferences, helping you to tailor your response. 
IMPORTANT: Your interaction with the customer is done through an automated system. You are not directly interacting with the customer, so avoid chitchat or follow up questions and focus PURELY on responding to the request with the necessary information. 
</important_background>

<core_responsibilities>
- Search and provide accurate information about songs, albums, artists, and playlists
- Offer relevant recommendations based on customer interests
- Handle music-related queries with attention to detail
- Help customers discover new music they might enjoy
- You are routed only when there are questions related to music catalog; ignore other questions. 
</core_responsibilities>

<guidelines>
1. Always perform thorough searches before concluding something is unavailable
2. If exact matches aren't found, try:
   - Checking for alternative spellings
   - Looking for similar artist names
   - Searching by partial matches
   - Checking different versions/remixes
3. When providing song lists:
   - Include the artist name with each song
   - Mention the album when relevant
   - Note if it's part of any playlists
   - Indicate if there are multiple versions
</guidelines>

Additional context is provided below: 

Prior saved user preferences: ${memory}

Message history is also attached.  
`;
}

// ============================================================================
// Nodes
// ============================================================================
//
// 什麼是 nodes？
// Nodes 是 LangGraph 的建構單元。每個 node 都是一個函數，會：
// 1. 將目前 state 作為輸入
// 2. 執行某些工作，例如呼叫 LLM、執行 tool、處理資料
// 3. 回傳要合併到 state 的更新
//
// 在 ReAct agent 中，通常有兩種類型的 nodes：
// - Agent node：呼叫 LLM 來決定下一步要做什麼
// - Tool node：執行 LLM 要求的 tools

// 設定資料庫
const db = await setupDatabase();

// 建立 tools
const musicTools = await createMusicTools(db);

// 將 tools 綁定到 model
// 這會告訴 LLM 有哪些 tools 可用，以及如何呼叫它們
const llmWithMusicTools = defaultModel.bindTools(musicTools);

// 使用預先建好的 ToolNode class 建立 tool node
// ToolNode 會自動執行 LLM 發出的 tool calls 並回傳結果
const musicToolNode = new ToolNode(musicTools);

// 建立 music assistant node
// 這是會呼叫 LLM 來決定下一步的「agent」node
async function musicAssistant(state: AgentState) {
  const memory = state.loadedMemory ?? "None";

  // 給 agent 的指令
  const musicAssistantPrompt = generateMusicAssistantPrompt(memory);

  // 使用 system prompt + conversation history 呼叫 model
  // LLM 會決定要直接回覆，或是使用 tool
  const response = await llmWithMusicTools.invoke([
    new SystemMessage(musicAssistantPrompt),
    ...state.messages,
  ]);

  // 回傳 state updates，messages 會附加到 state.messages
  return { messages: [response] };
}

// ============================================================================
// Conditional Edge
// ============================================================================
//
// 什麼是 conditional edges？
// Conditional edges 可依照 state 或邏輯來路由 graph flow。
// 它們是實作 ReAct loop 的關鍵。
//
// 此函數會判斷 agent 應該：
// - "continue"：前往 tool node（如果 LLM 要求 tool calls）
// - "end"：停止執行（如果 LLM 已提供最終答案）

function shouldContinue(state: AgentState): "continue" | "end" {
  const messages = state.messages;
  const lastMessage = messages.at(-1);

  // 如果有 tool calls，繼續到 tool node
  if (
    AIMessage.isInstance(lastMessage) &&
    lastMessage.tool_calls &&
    lastMessage.tool_calls.length > 0
  ) {
    return "continue";
  }
  // 如果沒有 tool calls，結束 graph
  return "end";
}

// ============================================================================
// Graph 建立
// ============================================================================
//
// 建置 GRAPH
// 現在使用 StateGraph 將所有部分串接起來。
//
// FLOW：
// START → music_assistant → [conditional] → music_tool_node → music_assistant → END
//                                      ↓
//                                     END（如果不需要 tools）
//
// 這會建立 ReAct loop：
// 1. Agent 思考並決策（music_assistant）
// 2. 如果需要 tools，就執行它們（music_tool_node）
// 3. 帶著 tool results 回到 agent
// 4. Agent 產生最終 response → END
//
// MEMORY COMPONENTS：
// - checkpointer：儲存 conversation history（短期記憶）
// - inMemoryStore：儲存 user preferences（長期記憶）

console.log("🎵 Creating Music Catalog Subagent...");

// 初始化 memory stores
const checkpointer = new MemorySaver();
const inMemoryStore = new InMemoryStore();

// 使用 StateGraph 建立 workflow
const musicWorkflow = new StateGraph(AgentState)
  // 加入 nodes，也就是實際執行工作的函數
  .addNode("music_assistant", musicAssistant)
  .addNode("music_tool_node", musicToolNode)
  
  // 加入 entry point，一律從 agent 開始
  .addEdge(START, "music_assistant")
  
  // 加入從 agent 出發的 conditional routing
  .addConditionalEdges("music_assistant", shouldContinue, {
    continue: "music_tool_node",  // 如果需要 tools，就執行它們
    end: END,                       // 如果不需要 tools，就完成
  })
  
  // Tools 執行後，一律回到 agent 處理結果
  .addEdge("music_tool_node", "music_assistant");

// 將 graph 編譯成可執行物件
// 編譯會驗證 graph 結構，並讓它準備好執行
export const graph = musicWorkflow.compile({
  checkpointer,      // 啟用 conversation persistence
  store: inMemoryStore,  // 啟用 long-term memory storage
});

console.log("✅ Music Catalog Subagent created successfully!");
