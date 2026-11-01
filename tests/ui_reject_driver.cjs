// Runs the script of the page the service serves, against that live service, through a stub DOM.
// Clicks reach the page's own listeners, and every request is a real fetch. See test_ui_reject.py.
"use strict";
const vm = require("node:vm");

const [base, token, rivalToken, firstReason, secondReason] = process.argv.slice(2);

class Element {
  value = "";
  innerHTML = "";
  textContent = "";
  className = "";
  disabled = false;
  listeners = {};
  addEventListener(type, listener) { this.listeners[type] = listener; }
  scrollIntoView() {}
}

const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id, new Element());
  return elements.get(id);
}

async function settled(button) {
  const deadline = Date.now() + 30000;
  while (button.disabled) {
    if (Date.now() > deadline) throw new Error("the page did not finish its action");
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
}

async function clickRow(act, id) {
  const button = new Element();
  button.dataset = {act, id};
  element("approvals").listeners.click({target: {closest: () => button}});
  await settled(button);
}

const rejectable = () =>
  [...element("approvals").innerHTML.matchAll(/data-act="reject" data-id="([^"]+)"/g)].map((m) => m[1]);

async function main() {
  const page = await (await fetch(`${base}/`)).text();
  const script = page.slice(page.lastIndexOf("<script>") + "<script>".length, page.lastIndexOf("</script>"));
  vm.runInNewContext(script, {
    document: {getElementById: element},
    fetch: (path, options) => fetch(`${base}${path}`, options),
    AbortSignal,
    setInterval: () => 0,
    clearInterval: () => {},
  });

  element("token").value = token;
  element("login").listeners.submit({preventDefault() {}});
  await settled(element("connect"));
  const [first, second, raced] = rejectable();
  if (!raced) throw new Error(`three proposals expected: ${element("error").innerHTML}`);

  element("reason").value = firstReason;
  await clickRow("reject", first);
  const afterRejecting = element("reason").value;

  await clickRow("reject", second);
  const errorWithoutTyping = element("error").innerHTML;

  // Another approver gets there first while this row still offers Reject, so this one is refused
  // and the typed reason has not been used.
  const approved = await fetch(`${base}/v1/proposals/${raced}/approve`, {
    method: "POST",
    headers: {"Authorization": `Bearer ${rivalToken}`, "Content-Type": "application/json"},
    body: "{}",
  });
  if (!approved.ok) throw new Error(`the rival approval failed: ${await approved.text()}`);
  element("reason").value = secondReason;
  await clickRow("reject", raced);
  const afterRefusal = element("reason").value;
  const errorOnRefusal = element("error").innerHTML;

  await clickRow("reject", second);

  process.stdout.write(JSON.stringify({
    first, second, afterRejecting, errorWithoutTyping, afterRefusal, errorOnRefusal,
  }));
}

main().then(() => process.exit(0), (error) => {
  process.stderr.write(`${error.stack}\n`);
  process.exit(1);
});
