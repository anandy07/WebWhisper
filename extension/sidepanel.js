const API = "http://127.0.0.1:8000/api/v1";
const $ = (id) => document.getElementById(id);
// One session per panel: re-analyzing replaces this panel's old page index.
const sessionId = crypto.randomUUID();
// True once a page is indexed; a failed re-analyze keeps the previous index.
let hasIndex = false;

async function post(path, body) {
  let res;
  try {
    res = await fetch(`${API}/${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch {
    throw new Error("Backend not running. Start it with: python main.py");
  }
  if (!res.ok) {
    const { detail } = await res.json().catch(() => ({}));
    // Validation errors (422) arrive as a list; show the first message.
    const message =
      typeof detail === "string"
        ? detail
        : detail?.[0]?.msg?.replace(/^Value error, /, "");
    throw new Error(message || `Request failed (${res.status})`);
  }
  return res;
}

// Runs inside the webpage: groups visible text under its nearest heading.
// Walks text nodes so text in divs and in list items with nested lists is kept.
function extractPage() {
  const root = document.querySelector("main, article") || document.body;
  // Site chrome is skipped, but an article's own <header> (title, intro) is kept.
  const skip =
    "nav,footer,aside,header:not(article header,main header),script,style,noscript,template";
  const block =
    "h1,h2,h3,h4,h5,h6,p,li,pre,blockquote,td,th,dd,dt,figcaption,caption,div,section,article,main";
  const sources = [];
  let current = { heading: null, text: "" };
  let lastBlock = null;
  const flush = () => current.text.trim() && sources.push(current);
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  for (let node; (node = walker.nextNode()); ) {
    const parent = node.parentElement;
    if (!node.data.trim() || !parent || parent.closest(skip)) continue;
    if (!parent.checkVisibility()) continue;
    const el = parent.closest(block) || root;
    if (/^H\d$/.test(el.tagName)) {
      if (el !== lastBlock) {
        flush();
        current = { heading: el.innerText.trim().slice(0, 500), text: "" };
      }
      lastBlock = el;
      continue;
    }
    let text = parent.closest("pre") ? node.data : node.data.replace(/\s+/g, " ");
    if (el !== lastBlock) text = "\n\n" + text.trimStart();
    else if (node.previousSibling?.nodeName === "BR") text = "\n" + text.trimStart();
    current.text += text;
    lastBlock = el;
  }
  flush();
  if (!sources.length) sources.push({ heading: null, text: root.innerText });
  return {
    page_url: location.href,
    page_title: (document.title || location.href).slice(0, 500),
    sources,
  };
}

// Strip markdown the model may still emit; the panel shows plain text.
const plain = (md) =>
  md
    .replace(/(\*\*|__|`)(.+?)\1/g, "$2")
    .replace(/^#{1,6}\s+/gm, "")
    .replace(/^\s*[*-]\s+/gm, "• ");

function addMessage(cls, text) {
  $("messages").querySelector(".note")?.remove();
  const div = document.createElement("div");
  div.className = `msg ${cls}`;
  div.textContent = text;
  $("messages").append(div);
  div.scrollIntoView({ block: "end" });
  return div;
}

function showNote(text) {
  const note = document.createElement("p");
  note.className = "note";
  note.textContent = text;
  $("messages").replaceChildren(note);
}

function setStatus(state, text) {
  $("page").dataset.state = state;
  $("status").textContent = text;
  $("status").title = text;
}

function setReady(ready) {
  $("question").disabled = !ready;
  $("send").disabled = !ready;
}

$("theme").onclick = toggleTheme;

$("analyze").onclick = async () => {
  setReady(false);
  $("analyze").disabled = true;
  setStatus("loading", "Analyzing...");
  try {
    const [tab] = await chrome.tabs.query({
      active: true,
      currentWindow: true,
    });
    const [{ result }] = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: extractPage,
    });
    await post("index", { session_id: sessionId, ...result });
    hasIndex = true;
    showNote("Ready. Ask a question below.");
    setStatus("ready", result.page_title);
    setReady(true);
    $("question").focus();
  } catch (err) {
    setStatus(
      "error",
      err.message.includes("Cannot access")
        ? "Chrome blocks reading this page."
        : `Failed: ${err.message}`,
    );
    // The backend still holds the last indexed page; keep chat usable.
    setReady(hasIndex);
  }
  $("analyze").disabled = false;
};

$("ask").onsubmit = async (e) => {
  e.preventDefault();
  const question = $("question").value.trim();
  if (!question) return;
  $("question").value = "";
  addMessage("user", question);
  const bot = addMessage("bot pending", "Thinking...");
  setReady(false);
  try {
    const res = await post("chat", { session_id: sessionId, question });
    let buffer = "",
      answer = "";
    for await (const chunk of res.body.pipeThrough(new TextDecoderStream())) {
      buffer += chunk;
      const events = buffer.split("\n\n");
      buffer = events.pop();
      for (const raw of events) {
        const event = JSON.parse(raw.replace(/^data: /, ""));
        if (event.type === "answer_delta") {
          bot.classList.remove("pending");
          bot.textContent = plain((answer += event.content));
        }
        if (event.type === "error") {
          bot.classList.remove("pending");
          bot.classList.add("error");
          bot.textContent = event.message;
        }
        if (event.type === "citations" && event.citations.length) {
          const sources = document.createElement("div");
          sources.className = "sources";
          sources.textContent = "Sources: ";
          for (const name of new Set(
            event.citations.map((c) => c.heading || c.page_title),
          )) {
            const tag = document.createElement("span");
            tag.textContent = name;
            tag.title = name;
            sources.append(tag);
          }
          bot.append(sources);
        }
      }
      bot.scrollIntoView({ block: "end" });
    }
    if (bot.classList.contains("pending")) {
      bot.classList.remove("pending");
      bot.textContent = "No answer received.";
    }
  } catch (err) {
    bot.classList.remove("pending");
    bot.classList.add("error");
    bot.textContent = `Error: ${err.message}`;
  }
  setReady(true);
  $("question").focus();
};
