// Chat page helpers: thinking blocks, the JSON-lines reader and the request body.
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const html=fs.readFileSync(path.join(__dirname,'../chat_ui.html'),'utf8');
const source=html.match(/<script>([\s\S]*?)<\/script>/)[1];
const context=vm.createContext({});
vm.runInContext(source,context);  // no document: the page does not start
const run=code=>vm.runInContext(code,context);

test('thinking blocks are split from the answer, also while still streaming',()=>{
  assert.deepEqual(JSON.parse(JSON.stringify(run(`splitThinking('<think>\\nplan it\\n</think>\\n\\nHello')`))),{thinking:'plan it',answer:'Hello'});
  assert.deepEqual(JSON.parse(JSON.stringify(run(`splitThinking('<think>half a thou')`))),{thinking:'half a thou',answer:''});
  assert.deepEqual(JSON.parse(JSON.stringify(run(`splitThinking('plain answer')`))),{thinking:'',answer:'plain answer'});
});

test('JSON lines split across network chunks are reassembled',()=>{
  const events=[];
  context.sink=e=>events.push(e);
  run(`const r=lineReader(sink);r.feed('{"type":"progress","event":{"pha');r.feed('se":"download"}}\\n{"type":"chunk"');r.feed(',"chunk":{}}\\n{"type":"done"}');r.flush();`);
  assert.deepEqual(events.map(e=>e.type),['progress','chunk','done']);
  assert.equal(events[0].event.phase,'download');
});

test('requests carry the system prompt, drop old thinking and bound the options',()=>{
  const body=JSON.parse(JSON.stringify(run(`chatRequest([
    {role:'user',content:'hi'},
    {role:'assistant',content:'<think>long reasoning</think>Hello!'},
    {role:'error',content:'Failed: x'},
    {role:'user',content:'again'}],{system:' Be brief. ',temperature:'0.3',maxTokens:'512'})`)));
  assert.deepEqual(body.messages,[{role:'system',content:'Be brief.'},{role:'user',content:'hi'},
    {role:'assistant',content:'Hello!'},{role:'user',content:'again'}]);
  assert.equal(body.temperature,0.3);
  assert.equal(body.max_tokens,512);
  const loose=JSON.parse(JSON.stringify(run(`chatRequest([{role:'user',content:'x'}],{system:'',temperature:'9',maxTokens:'abc'})`)));
  assert.deepEqual(Object.keys(loose),['messages']);
});

test('both languages define every label the page uses',()=>{
  const keys=[...html.matchAll(/data-tp?="([a-zA-Z]+)"/g)].map(m=>m[1]);
  for(const lang of ['zh','en'])for(const key of keys)assert.ok(run(`TEXT.${lang}`)[key],lang+'.'+key);
  assert.equal(run('gb(23214831232)'),'23.2 GB');
});
