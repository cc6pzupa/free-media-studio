// The Android download bridge: <a download> clicks reach CCAndroid intact.
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const source=fs.readFileSync(path.join(__dirname,'../android/app/src/main/assets/cc_bridge.js'),'utf8');

function page({fetch,begin=()=>1}={}){
  const saved=[],failures=[],aborted=[];let listener=null,nativeClicks=0;
  const files=new Map();
  const CCAndroid={
    begin(name,mime){const id=begin(name,mime);if(id>=0)files.set(id,{name,mime,parts:[]});return id},
    chunk(id,b64){files.get(id).parts.push(Buffer.from(b64,'base64'));return true},
    finish(id){const f=files.get(id);saved.push({name:f.name,mime:f.mime,bytes:Buffer.concat(f.parts)});return true},
    abort(id){aborted.push(id)},
    failed(message){failures.push(message)},
  };
  class HTMLAnchorElement{
    constructor(){this.attributes={};this.href=''}
    setAttribute(k,v){this.attributes[k]=String(v)}
    getAttribute(k){return k in this.attributes?this.attributes[k]:null}
    hasAttribute(k){return k in this.attributes}
    closest(selector){return selector==='a[download]'&&this.hasAttribute('download')?this:null}
    click(){nativeClicks++}
  }
  const urls=new Map();let serial=0;
  const URLShim={createObjectURL(blob){const u='blob:test/'+ ++serial;urls.set(u,blob);return u},revokeObjectURL(u){urls.delete(u)}};
  const context=vm.createContext({Blob,Uint8Array,String,Error,Map,setTimeout:()=>0,btoa,
    fetch:fetch||(async href=>{if(!urls.has(href))throw new Error('revoked');return {ok:true,blob:async()=>urls.get(href)}}),
    URL:URLShim,HTMLAnchorElement,CCAndroid,
    document:{addEventListener(type,fn){if(type==='click')listener=fn}}});
  context.window=context;
  vm.runInContext(source,context);
  return {context,saved,failures,aborted,listener:()=>listener,nativeClicks:()=>nativeClicks};
}
const settle=()=>new Promise(resolve=>setTimeout(resolve,20));

test('a large blob saved by click() survives an immediate revoke and splits into exact chunks',async()=>{
  const p=page();
  const bytes=Buffer.alloc(2*1024*1024+5);for(let i=0;i<bytes.length;i++)bytes[i]=(i*7+3)&255;
  p.context.payload=new Blob([bytes],{type:'application/zip'});
  vm.runInContext(`const a=new HTMLAnchorElement();a.href=URL.createObjectURL(payload);a.setAttribute('download','cc-media-batch.zip');a.click();URL.revokeObjectURL(a.href);`,p.context);
  await settle();
  assert.deepEqual(p.failures,[]);
  assert.equal(p.saved.length,1);
  assert.equal(p.saved[0].name,'cc-media-batch.zip');
  assert.equal(p.saved[0].mime,'application/zip');
  assert.ok(p.saved[0].bytes.equals(bytes));
  assert.equal(p.nativeClicks(),0);
});

test('clicking a static download link is intercepted; plain links are left alone',async()=>{
  const p=page({fetch:async href=>({ok:true,blob:async()=>new Blob(['notebook'],{type:'application/json'})})});
  const link=new p.context.HTMLAnchorElement();link.href='http://127.0.0.1:8787/kaggle_standalone.ipynb';link.setAttribute('download','kaggle_standalone.ipynb');
  let prevented=false;
  p.listener()({target:link,preventDefault(){prevented=true},stopPropagation(){}});
  await settle();
  assert.ok(prevented);
  assert.equal(p.saved[0].bytes.toString(),'notebook');
  const plain=new p.context.HTMLAnchorElement();plain.href='https://www.kaggle.com/';
  prevented=false;p.listener()({target:plain,preventDefault(){prevented=true},stopPropagation(){}});
  assert.equal(prevented,false);
});

test('storage and HTTP failures are reported, never half-saved',async()=>{
  const denied=page({begin:()=>-1});
  denied.context.payload=new Blob(['x']);
  vm.runInContext(`const a=new HTMLAnchorElement();a.href=URL.createObjectURL(payload);a.setAttribute('download','x.png');a.click();`,denied.context);
  await settle();
  assert.equal(denied.saved.length,0);
  assert.match(denied.failures[0],/storage/);
  const missing=page({fetch:async()=>({ok:false,status:404})});
  const link=new missing.context.HTMLAnchorElement();link.href='http://127.0.0.1:8787/gone';link.setAttribute('download','');
  link.click();await settle();
  assert.match(missing.failures[0],/HTTP 404/);
  assert.equal(missing.saved.length,0);
});
