const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const source=fs.readFileSync(require('node:path').join(__dirname,'../video_ui.html'),'utf8').match(/<script>([\s\S]*?)<\/script>/)[1];
const tick=()=>new Promise(resolve=>setImmediate(resolve));
const models=[{id:'wan21-t2v-1.3b',default_steps:30,width:832,height:480,fps:16,tier:'balanced'},
 {id:'ltx-video-2b-distilled',default_steps:8,fixed_steps:true,width:768,height:448,fps:16,tier:'efficient'}];
const response=body=>({ok:true,json:async()=>body,blob:async()=>({})});
function page(fetch,saved){
 const elements=new Map(),storage=new Map(saved instanceof Map?saved:saved?[['cc.media.video.job.v1',saved]]:[]);
 const element=id=>{if(!elements.has(id))elements.set(id,{style:{},value:id==='lang'?'zh':'',append(){},hidden:true});return elements.get(id)};
 const context=vm.createContext({fetch,console,Promise,crypto:{randomUUID:()=> 'request-id'},
  setTimeout:()=>1,clearTimeout(){},URL:{createObjectURL:()=> 'blob:video',revokeObjectURL(){}},
  document:{documentElement:{},getElementById:element,createElement:()=>({})},
  localStorage:{getItem:k=>storage.get(k),setItem:(k,v)=>storage.set(k,v),removeItem:k=>storage.delete(k)}});
 vm.runInContext(source,context);return {elements,storage};
}
test('video refresh retrieves the original MP4 without submitting another task',async()=>{
 const calls=[],id='a'.repeat(32);
 const p=page(async(url,options)=>{calls.push([url,options]);
  if(url==='/video/models')return response({models});
  if(url.endsWith('/content'))return response({});
  return response({state:'complete',usage:{model_id:'wan21-t2v-1.3b',frames:33,fps:16,parallel_strategy:'cfg_parallel',request_seconds:99}});
 },id);
 for(let i=0;i<8;i++)await tick();
 assert.deepEqual(calls.map(x=>x[0]),['/video/models','/video/jobs/'+id,'/video/jobs/'+id+'/content']);
 assert.ok(calls.every(x=>x[1].method==='GET'));
 assert.equal(p.elements.get('video').src,'blob:video');assert.equal(p.elements.get('download').hidden,false);
 assert.equal(p.storage.size,0);
});
test('video model defaults and download percentage stay visible',async()=>{
 const p=page(async url=>url==='/video/models'?response({models}):response({state:'running',phase:'download',percent:37,message:'Downloading'}),'b'.repeat(32));
 for(let i=0;i<6;i++)await tick();
 assert.equal(p.elements.get('bar').style.width,'37%');
 p.elements.get('model').value='ltx-video-2b-distilled';p.elements.get('model').onchange();
 assert.equal(p.elements.get('steps').value,8);assert.equal(p.elements.get('steps').disabled,true);
});
test('video sampling displays real step progress and video wording',async()=>{
 const p=page(async url=>url==='/video/models'?response({models}):response({state:'running',phase:'generating',step:15,steps:30,message:'正在生成图片。'}),'c'.repeat(32));
 for(let i=0;i<6;i++)await tick();
 assert.equal(p.elements.get('bar').style.width,'50%');
 assert.match(p.elements.get('status').textContent,/正在生成视频.*15\/30/);
});

test('temporary backend outage keeps the video job and never submits again',async()=>{
 const calls=[],id='d'.repeat(32);
 const p=page(async(url,options)=>{calls.push([url,options]);return url==='/video/models'?response({models}):{ok:false,status:503,json:async()=>({error:{message:'Worker unavailable'}})}},id);
 for(let i=0;i<6;i++)await tick();
 assert.equal(p.storage.get('cc.media.video.job.v1'),id);assert.equal(p.elements.get('generate').disabled,true);
 assert.ok(calls.every(x=>x[1].method==='GET'));
});
test('missing video job releases controls without submitting a replacement',async()=>{
 const calls=[],id='e'.repeat(32);
 const p=page(async(url,options)=>{calls.push([url,options]);return url==='/video/models'?response({models}):{ok:false,status:404,json:async()=>({error:{message:'Not found'}})}},id);
 for(let i=0;i<6;i++)await tick();
 assert.equal(p.storage.size,0);assert.equal(p.elements.get('generate').disabled,false);
 assert.match(p.elements.get('status').textContent,/原任务 ID/);assert.ok(calls.every(x=>x[1].method==='GET'));
});
test('lost submission response survives refresh and resumes the same request',async()=>{
 const payloads=[],id='f'.repeat(32);let lost=true;
 const fetch=async(url,options)=>{
  if(url==='/video/models')return response({models});
  if(options.method==='POST'){payloads.push(JSON.parse(options.body));if(lost){lost=false;throw Error('Response lost')}return response({job_id:id})}
  if(url.endsWith('/content'))return response({});
  return response({state:'complete',usage:{model_id:'wan21-t2v-1.3b',model_reused:true}});
 };
 const first=page(fetch);for(let i=0;i<6;i++)await tick();
 for(const [id,value] of [['prompt','anime catgirl'],['seed','42'],['frames','33']])first.elements.get(id).value=value;
 await first.elements.get('generate').onclick();
 assert.equal(payloads.length,1);assert.ok(first.storage.has('cc.media.video.pending.v1'));
 assert.equal(first.elements.get('prompt').disabled,true);
 const next=page(fetch,first.storage);for(let i=0;i<6;i++)await tick();
 assert.equal(payloads.length,1);assert.equal(next.elements.get('prompt').value,'anime catgirl');
 assert.match(next.elements.get('generate').textContent,/恢复/);
 await next.elements.get('generate').onclick();
 assert.deepEqual(payloads[0],payloads[1]);assert.equal(new Set(payloads.map(p=>p.request_id)).size,1);
 assert.equal(next.storage.size,0);assert.equal(next.elements.get('download').hidden,false);
 next.elements.get('lang').value='en';next.elements.get('lang').onchange();
 assert.equal(next.elements.get('status').textContent,'Video ready');assert.match(next.elements.get('meta').textContent,/Model reused/);
 assert.doesNotMatch(next.elements.get('meta').textContent,/undefined/);
});
