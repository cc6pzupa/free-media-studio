const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../easy_ui.html'),'utf8').match(/<script>([\s\S]*?)<\/script>/)[1];
const tick=()=>new Promise(resolve=>setImmediate(resolve));
function page(fetch,previous){
    const elements=new Map();
    const storage=new Map(previous?[['kaggle-qwen-pending-job',previous]]:[]);
    const context=vm.createContext({fetch,Date,Promise,setTimeout:fn=>setImmediate(fn),
        document:{getElementById(id){if(!elements.has(id))elements.set(id,{style:{},value:'',disabled:false});return elements.get(id)}},
        sessionStorage:{getItem:k=>storage.get(k)||null,setItem:(k,v)=>storage.set(k,v),removeItem:k=>storage.delete(k)}});
    vm.runInContext(source,context);
    return {elements,storage};
}
const response=(body,status=200)=>({ok:status===200,status,json:async()=>body});
const config=active=>response({configured:true,user:'owner',dataset:'owner/assets',endpoint:'local',api_key:'test',active_job_id:active});
const complete=()=>response({state:'complete',result:{data:[{b64_json:'cG5n'}],usage:{request_seconds:12}}});
test('refresh restores the active task without submitting another',async()=>{
    const calls=[];
    const p=page(async(url,options)=>{calls.push([url,options]);return url==='/configuration'?config('existing'):complete()});
    for(let i=0;i<5;i++)await tick();
    assert.deepEqual(calls.map(c=>c[0]),['/configuration','/jobs/existing']);
    assert.ok(calls.every(c=>!c[1].method));
    assert.equal(p.elements.get('result').src,'data:image/png;base64,cG5n');
    assert.equal(p.storage.size,0);
    assert.equal(p.elements.get('generate').disabled,false);
});
test('refresh retrieves a task that finished while the page was closed',async()=>{
    const calls=[];
    const p=page(async url=>{calls.push(url);return url==='/configuration'?config(null):complete()},'finished');
    for(let i=0;i<5;i++)await tick();
    assert.deepEqual(calls,['/configuration','/jobs/finished']);
    assert.equal(p.elements.get('save').style.display,'inline');
});
test('network interruption keeps task identity for the next refresh',async()=>{
    const p=page(async url=>{if(url==='/configuration')return config('active');throw Error('offline')});
    for(let i=0;i<5;i++)await tick();
    assert.equal(p.storage.get('kaggle-qwen-pending-job'),'active');
    assert.match(p.elements.get('status').textContent,/offline/);
});
test('server restart clears expired local task and never resubmits',async()=>{
    const calls=[];
    const p=page(async url=>{calls.push(url);return url==='/configuration'?config(null):response({},404)},'old');
    for(let i=0;i<5;i++)await tick();
    assert.deepEqual(calls,['/configuration','/jobs/old']);
    assert.equal(p.storage.size,0);
    assert.match(p.elements.get('status').textContent,/Kaggle/);
});
test('completed warm request explicitly reports that model was reused',async()=>{
    const p=page(async url=>{
        if(url==='/configuration')return config('warm');
        return response({state:'complete',result:{data:[{b64_json:'cG5n'}],usage:{backend:'session',model_reused:true,request_seconds:65}}});
    });
    for(let i=0;i<5;i++)await tick();
    assert.match(p.elements.get('status').textContent,/已复用现有模型/);
});
test('download progress reports actual bytes and percentage',async()=>{
    let count=0;
    const p=page(async url=>{
        if(url==='/configuration')return config('download');
        if(count++===0)return response({state:'running',phase:'download',message:'正在下载',downloaded_bytes:1e9,total_bytes:4e9,percent:25,file:'model.safetensors'});
        throw Error('pause test');
    });
    for(let i=0;i<7;i++)await tick();
    assert.equal(p.elements.get('progressBar').style.width,'25%');
    assert.match(p.elements.get('progressCaption').textContent,/1.00 \/ 4.00 GB/);
});

test('model selection uses trained defaults and includes the selected model in submission',async()=>{
    const calls=[];
    const p=page(async(url,options)=>{
        calls.push([url,options]);
        if(url==='/configuration')return config(null);
        if(url==='/jobs')return response({job_id:'model-job'});
        return complete();
    });
    for(let i=0;i<5;i++)await tick();
    const choice=p.elements.get('modelChoice');
    choice.value='sdxl-lightning';choice.onchange();
    assert.equal(p.elements.get('steps').value,4);
    assert.equal(p.elements.get('steps').disabled,true);
    p.elements.set('prompt',{value:'adult catgirl',style:{}});
    p.elements.set('seed',{value:'42',style:{}});
    await p.elements.get('generate').onclick();
    const payload=JSON.parse(calls.find(([url])=>url==='/jobs')[1].body);
    assert.equal(payload.model_id,'sdxl-lightning');assert.equal(payload.steps,4);
    choice.value='z-image-turbo';choice.onchange();
    assert.equal(p.elements.get('steps').value,9);assert.equal(p.elements.get('steps').disabled,false);
    choice.value='qwen-image-2.1';choice.onchange();
    assert.equal(p.elements.get('steps').value,20);
});

test('saved acceleration selection is restored and submitted with account configuration',async()=>{
    const calls=[];
    const p=page(async(url,options)=>{
        calls.push([url,options]);
        if(url==='/configuration')return response({configured:true,user:'owner',dataset:'owner/assets',acceleration:'split'});
        return response({ok:true,endpoint:'local',api_key:'local'});
    });
    for(let i=0;i<5;i++)await tick();
    assert.equal(p.elements.get('acceleration').value,'split');
    p.elements.get('acceleration').value='auto';
    for(const [id,value] of [['token','key'],['keepAlive','15'],['jupyterUrl','']])p.elements.set(id,{value,style:{}});
    p.elements.set('remember',{checked:true,style:{}});
    await p.elements.get('configure').onclick();
    assert.equal(JSON.parse(calls.find(([url])=>url==='/configure')[1].body).acceleration,'auto');
});

test('completed image shows actual dual strategy and model transfer timing',async()=>{
    const p=page(async url=>url==='/configuration'?config('dual'):response({state:'complete',result:{data:[{b64_json:'cG5n'}],usage:{backend:'account',model_reused:true,request_seconds:52,parallel_strategy:'cfg_parallel',migration_seconds:13}}}));
    for(let i=0;i<5;i++)await tick();
    assert.match(p.elements.get('resultMeta').textContent,/CFG 双卡并行/);
    assert.match(p.elements.get('status').textContent,/迁移 13/);
});
