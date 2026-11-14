const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const source=fs.readFileSync(require('node:path').join(__dirname,'../gallery_ui.html'),'utf8').match(/<script>([\s\S]*?)<\/script>/)[1];
const tick=()=>new Promise(resolve=>setImmediate(resolve));
const item=(id,kind='video')=>({job_id:id,created:1000,media_type:kind,favorite:false,available:true,params:{prompt:'<img src=x onerror=alert(1)>',model_id:'wan22-ti2v-5b',seed:42,steps:30},usage:{}});
const response=body=>({ok:true,json:async()=>body,blob:async()=>({})});
function page(fetch){
 const elements=new Map(),revoked=[],observed=[];
 function node(tag='div'){return {tag,children:[],value:'',hidden:false,disabled:false,options:[],setAttribute(k,v){this[k]=v},append(...n){this.children.push(...n)},replaceChildren(...n){this.children=n},querySelector(){return null},showModal(){this.open=true},close(){this.open=false;this.onclose?.()}}}
 const element=id=>{if(!elements.has(id)){const n=node();if(id==='kind'){n.value='all';n.options=['all','image','video'].map(value=>({value}))}elements.set(id,n)}return elements.get(id)};
 let serial=0;
 const context=vm.createContext({fetch,console,Date,URLSearchParams,Promise,confirm:()=>true,navigator:{clipboard:{writeText:async()=>{}}},URL:{createObjectURL:()=> 'blob:'+ ++serial,revokeObjectURL:u=>revoked.push(u)},IntersectionObserver:class{observe(n){observed.push(n)}disconnect(){}unobserve(){}},localStorage:{getItem:()=>null,setItem(){}},document:{documentElement:{},getElementById:element,createElement:node},window:{addEventListener(){}}});
 vm.runInContext(source,context);return {elements,revoked,observed,context};
}
test('gallery list never submits generation and renders remote prompts as text',async()=>{
 const calls=[];const p=page(async(url,options)=>{calls.push([url,options]);return response({items:[item('a'.repeat(32))],total:1,next_offset:null})});
 for(let i=0;i<4;i++)await tick();
 assert.ok(calls.every(c=>c[1].method==='GET'));
 const card=p.elements.get('grid').children[0];assert.equal(card.children[1].children[0].textContent,'<img src=x onerror=alert(1)>');
 assert.equal(p.elements.get('more').hidden,true);
 assert.ok(calls[0][1].headers['X-Local-Session']);
});
test('video loads only when opened and blob URLs are released on close',async()=>{
 const calls=[];const p=page(async(url)=>{calls.push(url);return response(url.startsWith('/gallery/media/')?{}:{items:[item('b'.repeat(32))],total:1,next_offset:null})});
 for(let i=0;i<4;i++)await tick();assert.equal(calls.length,1);
 await p.elements.get('grid').children[0].children[0].onclick();
 assert.equal(calls.length,2);assert.equal(p.elements.get('viewer').children[0].tag,'video');assert.equal(p.elements.get('download').hidden,false);
 p.elements.get('close').onclick();assert.deepEqual(p.revoked,['blob:1']);assert.equal(p.elements.get('download').hidden,true);
});
test('late old search result cannot replace a newer filter',async()=>{
 let old;const p=page(async()=>new Promise(resolve=>{old=resolve}));
 p.elements.get('kind').value='video';
 vm.runInContext("request=async()=>({json:async()=>({items:[],total:0,next_offset:null})})",p.context);
 p.elements.get('kind').onchange();for(let i=0;i<4;i++)await tick();
 old(response({items:[item('c'.repeat(32))],total:1,next_offset:null}));for(let i=0;i<4;i++)await tick();
 assert.equal(p.elements.get('grid').children.length,0);
});
test('pagination keeps lazy image loads from the previous page valid',async()=>{
 let n=0;const p=page(async url=>url.startsWith('/gallery/media/')?response({}):response({items:[item(String(++n).repeat(32),'image')],total:2,next_offset:n===1?1:null}));
 for(let i=0;i<4;i++)await tick();const first=p.observed[0];
 await p.elements.get('more').onclick();await first._load();
 assert.equal(first.children[0].tag,'img');assert.equal(p.elements.get('grid').children.length,2);
});
test('selection mode zips and deletes only the chosen works',async()=>{
 const calls=[];const ids=['a','b','c'].map(c=>c.repeat(32));
 const p=page(async(url,options)=>{calls.push([url,options]);
  if(url.startsWith('/gallery/items?'))return response({items:ids.map(id=>item(id,'image')),total:3,next_offset:null});
  return response({ok:true});
 });
 for(let i=0;i<4;i++)await tick();
 const grid=p.elements.get('grid');
 p.elements.get('select').onclick();
 assert.equal(p.elements.get('selectionBar').hidden,false);
 grid.children[0].children[0].onclick();grid.children[2].children[0].onclick();
 assert.equal(calls.filter(([url])=>url.startsWith('/gallery/media/')).length,0);  // selecting never opens the viewer
 assert.match(p.elements.get('selectedCount').textContent,/2/);
 assert.equal(grid.children[0].className,'card selected');assert.equal(grid.children[1].className,'card');
 await p.elements.get('zipSelected').onclick();
 const zip=calls.find(([url])=>url==='/gallery/zip');
 assert.deepEqual(JSON.parse(zip[1].body).job_ids,[ids[0],ids[2]]);
 await p.elements.get('deleteSelected').onclick();
 const deleted=calls.filter(([url,options])=>url.startsWith('/gallery/items/')&&options.method==='POST').map(([url,options])=>[url.slice(15),JSON.parse(options.body).action]);
 assert.deepEqual(deleted,[[ids[0],'delete'],[ids[2],'delete']]);
 assert.match(p.elements.get('status').textContent,/已删除 2 件/);
 p.elements.get('select').onclick();assert.equal(p.elements.get('selectionBar').hidden,true);
});
