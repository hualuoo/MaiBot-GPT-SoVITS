import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const root = new URL('../', import.meta.url);
const html = readFileSync(new URL('webui/setup.html', root), 'utf8');
const declaration = JSON.parse(readFileSync(new URL('webui.json', root), 'utf8'));
const scripts = [...html.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/gi)].map((match) => match[1]);
const source = scripts.join('\n');
const chinese = /[\u3400-\u9fff]/u;
const textOnly = (text) => text.replace(/<[^>]+>/g, '').trim();
const json = (value) => JSON.parse(JSON.stringify(value));

const expectedQueries = { ensure_connection: ['setup_ensure_connection', []], state: ['setup_state', []], catalog: ['setup_catalog', ['root']], job: ['setup_job', ['job_id']], audio: ['setup_audio', ['job_id', 'offset']], roles: ['setup_roles', []], check_config: ['setup_check_config', []], llm_tasks: ['setup_llm_tasks', []] };
const modelKeys = ['root', 'base_model', 'gpt_weights', 'sovits_weights', 'lora_weights', 'ref_audio_path', 'prompt_text', 'prompt_lang', 'text_lang'];
const parameterKeys = ['speed_factor','top_k','top_p','temperature','seed','text_split_method','parallel_infer','max_text_length','cooldown_seconds','send_text'];
const autoKeys = ['enabled','frequency_mode','probability_percent','every_n','min_interval_seconds','follow_emotion','keep_text'];
const autoDefaults = {enabled:false,frequency_mode:'probability',probability_percent:50,every_n:3,min_interval_seconds:5,follow_emotion:true,keep_text:false};
const expectedActions = { save_auto_voice: ['setup_save_auto_voice', autoKeys], save_parameters: ['setup_save_parameters', parameterKeys], save_base: ['setup_save_base', ['root','sovits_weights']], save_api: ['setup_save_api', ['base_url', 'timeout']], test_api: ['setup_test_api', []], save_models: ['setup_save_models', modelKeys], start_job: ['setup_start_job', ['kind', 'text']], cancel_job: ['setup_cancel_job', ['job_id']], set_mode: ['setup_set_mode', ['mode']], prepare_role: ['setup_prepare_role', ['role_id','emotion','emotions']], save_emotion_judge: ['setup_save_emotion_judge', ['llm_emotion','llm_task']], test_emotion: ['setup_test_emotion', ['text']] };

function validateArguments(binding, args) {
  assert.deepEqual(Object.keys(args).sort(), Object.keys(binding.parameters).sort(), '调用参数必须与声明完全一致');
  for (const [key, parameter] of Object.entries(binding.parameters)) {
    const value = args[key];
    if (parameter.type === 'string') { assert.equal(typeof value, 'string'); assert.ok(Array.from(value).length <= parameter.max_length); }
    if (parameter.type === 'boolean') assert.equal(typeof value, 'boolean');
    if (parameter.type === 'number') assert.ok(typeof value === 'number' && Number.isFinite(value));
    if (parameter.type === 'integer') assert.ok(Number.isSafeInteger(value));
    if (parameter.type === 'array') { assert.ok(Array.isArray(value)); assert.ok(value.length <= parameter.max_items); for (const item of value) { assert.equal(typeof item, parameter.items.type); assert.ok(Array.from(item).length <= parameter.items.max_length); } }
    if (parameter.minimum !== undefined) assert.ok(value >= parameter.minimum);
    if (parameter.maximum !== undefined) assert.ok(value <= parameter.maximum);
    if (parameter.choices) assert.ok(parameter.choices.includes(value));
  }
}

test('声明为协议二的单工作区自定义中文设置页', () => {
  assert.equal(declaration.schema_version, 2);
  assert.equal(declaration.workspace_title, 'GPT-SoVITS 语音设置');
  assert.deepEqual(declaration.required_capabilities, ['custom_pages','structured_arguments']);
  assert.equal(declaration.pages.length, 1);
  const page = declaration.pages[0];
  assert.equal(page.id, 'setup'); assert.equal(page.mode, 'custom'); assert.equal(page.placement, 'workspace');
  assert.equal(page.entry, 'webui/setup.html'); assert.equal(page.layout, 'document');
  for (const key of ['title', 'description']) assert.match(page[key], chinese);
  for (const [kind, expected] of [['queries', expectedQueries], ['actions', expectedActions]]) {
    assert.deepEqual(Object.keys(page[kind]).sort(), Object.keys(expected).sort());
    for (const [name, [api, keys]] of Object.entries(expected)) {
      assert.equal(page[kind][name].api, api);
      assert.deepEqual(Object.keys(page[kind][name].parameters).sort(), [...keys].sort());
      for (const parameter of Object.values(page[kind][name].parameters)) {
        assert.equal(parameter.required, true);
        if (parameter.type === 'string') assert.ok(Number.isInteger(parameter.max_length) && parameter.max_length > 0 && parameter.max_length <= 4000);
      }
    }
  }
});

test('参数范围精确、空参数查询不携带多余字段', () => {
  const { queries, actions } = declaration.pages[0];
  assert.deepEqual(queries.state.parameters, {}); assert.deepEqual(actions.test_api.parameters, {});
  assert.equal(queries.catalog.parameters.root.max_length, 1024);
  assert.deepEqual(Object.keys(queries.catalog.parameters), ['root']);
  assert.deepEqual(actions.save_base.parameters, {root:{type:'string', required:true, max_length:1024}, sovits_weights:{type:'string', required:true, max_length:1024}});
  assert.equal(queries.job.parameters.job_id.max_length, 64); assert.equal(queries.audio.parameters.job_id.max_length, 64);
  assert.deepEqual(queries.audio.parameters.offset, { type: 'integer', required: true, minimum: 0, maximum: 16777216 });
  assert.equal(actions.save_api.parameters.base_url.max_length, 2048);
  assert.deepEqual(actions.save_api.parameters.timeout, { type: 'number', required: true, minimum: 1, maximum: 240 });
  assert.deepEqual(actions.start_job.parameters.kind.choices, ['load_models', 'synthesize']);
  assert.equal(actions.start_job.parameters.text.max_length, 300);
  assert.equal(actions.save_models.parameters.base_model.max_length, 80);
  assert.equal(actions.save_models.parameters.prompt_text.max_length, 4000);
  for (const key of ['root', 'gpt_weights', 'sovits_weights', 'lora_weights', 'ref_audio_path']) assert.equal(actions.save_models.parameters[key].max_length, 1024);
  for (const key of ['prompt_lang', 'text_lang']) assert.equal(actions.save_models.parameters[key].max_length, 32);
});

test('内联脚本由 vm.Script 静态编译且只使用 MaiBot 桥接', () => {
  assert.equal(scripts.length, 1); new vm.Script(source, { filename: 'webui/setup.html' });
  assert.match(source, /window\.maibot/); assert.match(source, /sdk\.ready/);
  assert.match(source, /ui\.sdk\.query\(name, args\)/); assert.match(source, /ui\.sdk\.action\(name, args\)/);
  assert.doesNotMatch(source, /\bfetch\s*\(|\bXMLHttpRequest\b|\bWebSocket\b|\bEventSource\b|\bimport\s*\(/);
  assert.doesNotMatch(html, /<(?:script|link|iframe)[^>]+(?:src|href)\s*=/i);
  assert.doesNotMatch(source, /\.innerHTML\s*=|document\.write\s*\(/);
});

test('所有表单都有中文可访问标签和帮助，控件标识唯一', () => {
  assert.match(html, /<html lang="zh-CN">/); assert.match(html, /<meta name="viewport"/);
  const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]);
  assert.equal(new Set(ids).size, ids.length);
  for (const match of html.matchAll(/<(input|textarea|select)\b([^>]*)>/g)) {
    const id = match[2].match(/\bid="([^"]+)"/)?.[1]; assert.ok(id);
    const label = html.match(new RegExp('<label[^>]+for="' + id + '"[^>]*>([\\s\\S]*?)<\\/label>'));
    assert.ok(label, id + ' 缺少标签'); assert.match(textOnly(label[1]), chinese);
    assert.match(match[2], /aria-describedby="[^"]+"/);
    for (const help of match[2].match(/aria-describedby="([^"]+)"/)[1].split(' ')) assert.ok(ids.includes(help));
  }
  for (const match of html.matchAll(/<button\b[^>]*>([\s\S]*?)<\/button>/g)) assert.match(textOnly(match[1]), chinese);
  for (const match of html.matchAll(/\baria-label="([^"]+)"/g)) assert.match(match[1], chinese);
  assert.match(html, /aria-live="polite"/); assert.match(html, /focus-visible/); assert.match(html, /prefers-reduced-motion/);
});

test('原生卡片、亮暗主题、响应式和共享服务警告齐全', () => {
  for (const token of ['maibot-page', 'maibot-card', 'maibot-input', 'maibot-select', 'maibot-button', 'maibot-button-primary']) assert.ok(html.includes(token));
  for (const token of ['background', 'surface', 'foreground', 'muted', 'soft', 'border', 'primary', 'primary-foreground', 'danger', 'radius', 'font-body']) assert.ok(html.includes('--maibot-' + token));
  assert.match(html, /:root\.dark/); assert.match(html, /prefers-color-scheme:dark/); assert.match(html, /@media\(max-width:520px\)/);
  assert.match(html, /共享服务的全局权重/); assert.match(html, /保存设置不会加载模型/); assert.match(html, /远程服务/);
  for (const id of ['sovits-model-picker','gpt-model-picker','lora-model-picker']) assert.match(html, new RegExp('<select id="' + id + '"'));
  assert.doesNotMatch(html, /id="gpt-weights"|id="lora-weights"|id="gpt-options"/);
  assert.match(html, /<input id="base-model"[^>]*\breadonly\b/); assert.doesNotMatch(html, /id="sovits-weights"|id="sovits-options"|id="model-extra-dir"|id="save-api"/);
  assert.doesNotMatch(html, /<audio[^>]*\bautoplay\b/i); assert.doesNotMatch(source, /\.play\s*\(/);
});

class FakeElement {
  constructor(id = '', attrs = '') { this.id = id; this.attrs = new Map([...attrs.matchAll(/([\w-]+)="([^"]*)"/g)].map((m) => [m[1], m[2]])); this.value = this.attrs.get('value') || ''; this.textContent = ''; this.disabled = /\sdisabled(?:\s|$)/.test(' ' + attrs); this.hidden = /\shidden(?:\s|$)/.test(' ' + attrs); this.dataset = {}; this.listeners = new Map(); this.children = []; }
  addEventListener(name, listener) { const items = this.listeners.get(name) || []; items.push(listener); this.listeners.set(name, items); }
  click() { if (!this.disabled) this.emit('click'); }
  emit(name, extra = {}) { const event = {target:this,preventDefault() { this.defaultPrevented = true; },...extra}; for (const listener of this.listeners.get(name) || []) listener(event); return event; }
  setAttribute(name, value) { this.attrs.set(name, String(value)); }
  removeAttribute(name) { this.attrs.delete(name); if (name === 'src') delete this.src; }
  toggleAttribute(name, force) { if (force) this.attrs.set(name, ''); else this.attrs.delete(name); }
  get options() { return this.children; }
  replaceChildren(...children) { this.children = children; }
  append(child) { this.children.push(child); }
  focus() { this.focused = true; }
  pause() { this.pauseCalls = (this.pauseCalls || 0) + 1; }
  load() {}
}
function defaultState() {
  return { config: { plugin: { enabled: true }, api: { base_url: 'http://127.0.0.1:9880', timeout: 180, max_audio_bytes: 8388608 }, models: { root: 'C:/GPT-SoVITS', base_model: 'custom', gpt_weights: 'C:/模型/语义.ckpt', sovits_weights: 'C:/模型/声学.pth', lora_weights: '' }, voice: { ref_audio_path: '', prompt_text: '', prompt_lang: 'zh', text_lang: 'zh' }, role: {mode:'original',role_id:'',emotion:'',emotions:[]}, behavior: {} }, connection: { status: 'idle', message: '请测试接口' }, model_status: { status: 'idle', message: '请加载模型' }, role_status:{status:'idle',message:'请选择角色',profile:null}, job: null };
}
// 第一步已完成（接口本次测试成功）：用于测试后续步骤本身的行为。
function connectedState(state = defaultState()) { state.connection = {status:'success',message:'接口测试成功',checked_url:state.config.api.base_url.replace(/\/+$/, '')}; return withReference(state); }
// 普通模式第三步完成需要已保存的参考音频（放在 GPT-SoVITS 目录下的 maibot_refs）。
function withReference(state, path = 'C:/GPT-SoVITS/maibot_refs/平静.wav') { state.config.voice = {...state.config.voice, ref_audio_path:path}; return state; }
function applyRoleState(state) {
  const {role_id,emotion} = state.config.role;
  const emotions = state.config.role.emotions || [emotion];
  state.config.models = {...state.config.models,base_model:'custom',gpt_weights:'data/' + role_id + '/' + emotion + '.ckpt',sovits_weights:'data/_shared/base.pth',lora_weights:'data/' + role_id + '/' + emotion + '.pth'};
  state.config.voice = {ref_audio_path:'data/' + role_id + '/' + emotion + '.wav',prompt_text:'<img src=x onerror=alert(1)> 黍的' + emotion + '参考文字',prompt_lang:'zh',text_lang:'zh'};
  state.model_status = {status:'idle',message:'角色配置已更新，请手动加载'};
  state.role_status = {status:'success',message:'JSON 自动配置与资源校验完成',profile:{role_id,role_name:role_id,emotion,emotion_label:'情绪：' + emotion,models:json(state.config.models),voice:json(state.config.voice),assets:[{kind:'GPT_Lora',path:state.config.models.gpt_weights,size:65539,sha256:'b'.repeat(64),reused:true}],warnings:['共享底模已复用']}};
  state.role_status.emotions = [...emotions];
  state.role_status.profiles = Object.fromEntries(emotions.map(id => [id, {...json(state.role_status.profile), emotion:id, models:{...state.config.models,gpt_weights:'data/' + role_id + '/' + id + '.ckpt',lora_weights:'data/' + role_id + '/' + id + '.pth'},voice:{...state.config.voice,ref_audio_path:'data/' + role_id + '/' + id + '.wav'}}]));
}
function roleState(prepared = false) {
  const state = defaultState(); state.connection.status = 'success'; state.config.role = {mode:'role',role_id:'黍',emotion:'happy',emotions:['happy']};
  if (prepared) applyRoleState(state);
  return state;
}

test('角色桥接声明精确，网页不提供上传入口', () => {
  const {queries,actions} = declaration.pages[0];
  assert.deepEqual(queries.roles.parameters, {}); assert.ok(Object.keys(actions).length <= 20);
  assert.deepEqual(actions.set_mode.parameters.mode.choices,['original','role']);
  assert.deepEqual(actions.prepare_role.parameters.emotions,{type:'array',required:true,max_items:50,items:{type:'string',max_length:80}});
  assert.equal(actions.prepare_role.parameters.role_id.max_length,240); assert.equal(actions.prepare_role.parameters.emotion.max_length,80);
  assert.equal(Object.keys(actions).some(name => name.startsWith('upload')), false);
  assert.doesNotMatch(html, /type="file"|id="upload-/); assert.doesNotMatch(source, /upload_|uploadFiles|base64_data/);
  assert.doesNotMatch(source,/crypto\.subtle|\.digest\s*\(/);
});

test('默认原版模式，切换保存 backend 并回读手工快照，不自动加载', async () => {
  const f = fixture({state:connectedState()}); await f.flush(); await f.click('step-models');
  assert.equal(f.get('panel-models').hidden,false); assert.equal(f.get('role-card').hidden,true);
  const original = f.get('gpt-model-picker').value; await f.click('role-mode-btn');
  assert.equal(f.get('panel-models').hidden,true); assert.equal(f.get('role-card').hidden,false);
  assert.deepEqual(f.calls.findLast(call => call.name === 'set_mode').args,{mode:'role'});
  assert.equal(f.get('role-mode-btn').attrs.get('aria-selected'),'true');
  f.state.config.models.gpt_weights = 'data/黍/角色.ckpt'; await f.click('original-mode-btn');
  assert.equal(f.get('gpt-model-picker').value,original); assert.equal(f.get('panel-models').hidden,false);
  assert.equal(f.calls.some(call => call.name === 'start_job' || call.name === 'prepare_role'),false);
});

test('roles 无参列出五情绪，错误与特殊标签仅以文字显示', async () => {
  const f = fixture({query:name => name === 'roles' ? {roles:[{id:'黍',name:'<img src=x>黍',emotions:['happy','angry','surprised','neutral','sad'].map(id=>({id,label:id})),json_path:'data/黍/黍.json'}],errors:['另一个角色 JSON 无效']} : undefined});
  await f.flush(); await f.click('role-mode-btn'); await f.click('refresh-roles');
  assert.deepEqual(f.calls.findLast(call => call.name === 'roles').args,{});
  assert.equal(f.get('role-picker').options[0].textContent,'<img src=x>黍');
  assert.deepEqual(f.get('emotion-options').children.map(item=>item.children[0].value),['happy','angry','surprised','neutral','sad']);
  assert.deepEqual(f.get('emotion-picker').options.map(item=>item.value),['happy']);
  assert.match(f.get('roles-note').textContent,/JSON 无效/); assert.equal(f.calls.some(call=>call.name==='start_job'),false);
});

test('读取角色配置启动 prepare_role，成功自动回读 models voice 摘要但不自动加载', async () => {
  const f = fixture({state:roleState()}); await f.flush(); await f.click('step-models'); await f.click('apply-role');
  assert.deepEqual(f.calls.findLast(call => call.name === 'prepare_role').args,{role_id:'黍',emotion:'happy',emotions:['happy']});
  assert.equal(f.get('role-load-models').disabled,true); assert.match(f.get('job-title').textContent,/角色资源准备/);
  await f.tick();
  assert.equal(f.get('gpt-model-picker').value,'data/黍/happy.ckpt'); assert.equal(f.get('ref-audio-path').value,'data/黍/happy.wav');
  assert.equal(f.get('role-ref-text').value,f.state.config.voice.prompt_text); assert.equal(f.get('role-ref-text').readOnly,true);
  assert.match(f.get('role-status').textContent,/JSON 自动配置/); assert.match(f.get('role-status').textContent,/已校验并复用/); assert.match(f.get('role-status').textContent,/bbbbbbbb/);
  assert.equal(f.get('role-load-models').disabled,false); assert.equal(f.get('role-synthesize').disabled,true);
  assert.equal(f.calls.some(call => call.name === 'audio' || call.name === 'start_job'),false);
});

test('角色页加载与试听按钮委托原有流程，手工卡隐藏仍可用', async () => {
  const f = fixture({state:roleState(true)}); await f.flush(); await f.click('step-models');
  assert.equal(f.get('panel-models').hidden,true); await f.click('role-load-models');
  assert.deepEqual(f.calls.filter(call=>call.kind==='action').map(call=>call.name),['start_job']);
  assert.deepEqual(f.calls.findLast(call=>call.name==='start_job').args,{kind:'load_models',text:''}); await f.tick();
  assert.equal(f.get('role-synthesize').disabled,false); await f.click('role-synthesize');
  assert.equal(f.calls.findLast(call=>call.name==='start_job').args.kind,'synthesize');
  assert.equal(f.calls.filter(call=>call.name==='save_models').length,0);
});

test('改变角色或情绪立即失效 readiness，切回原选项与刷新也不能复活旧 profile', async () => {
  const state=roleState(true); state.model_status.status='success'; const f=fixture({state}); await f.flush();
  assert.equal(f.get('role-synthesize').disabled,false);
  await f.checkEmotion('sad',true); await f.input('emotion-picker','sad','change'); assert.equal(f.get('role-load-models').disabled,true); assert.equal(f.get('role-ref-text').value,'');
  await f.input('emotion-picker','happy','change'); await f.click('refresh-state');
  assert.equal(f.get('role-synthesize').disabled,true); assert.equal(f.get('role-load-models').disabled,true); assert.match(f.get('role-status').textContent,/重新读取/);
  await f.input('role-picker','另一个角色','change'); assert.equal(f.get('role-load-models').disabled,true);
});

test('准备期间改变情绪不会被旧任务自动回读误判当前角色就绪', async () => {
  const f=fixture({state:roleState()}); await f.flush(); await f.click('apply-role'); await f.checkEmotion('sad',true); await f.input('emotion-picker','sad','change'); await f.tick();
  assert.equal(f.get('emotion-picker').value,'sad'); assert.equal(f.get('role-load-models').disabled,true); assert.equal(f.get('role-ref-text').value,'');
  assert.match(f.get('role-status').textContent,/准备期间.*修改/); assert.equal(f.calls.some(call=>call.name==='audio'),false);
});

test('prepare_role 恢复轮询与 error/cancelled 不当作模型成功或音频', async () => {
  for (const status of ['error','cancelled']) {
    const state=roleState(); state.job={id:'恢复准备',kind:'prepare_role',status:'running',message:'正在校验'};
    const f=fixture({state,query:name=>name==='job'?{...state.job,status,message:'资源校验未完成',audio_bytes:100}:undefined}); await f.flush();
    assert.equal([...f.timers.values()].filter(timer=>timer.ms===1000).length,1); await f.tick();
    assert.equal(f.get('role-load-models').disabled,true); assert.equal(f.calls.some(call=>call.name==='audio'),false); assert.equal(f.get('synthesize').disabled,true);
  }
});

test('角色从列表移除后不能继续使用旧 profile，重新出现也不自动就绪', async () => {
  let removed=false; const state=roleState(true); state.model_status.status='success';
  const f=fixture({state,query:name=>name==='roles'&&removed?{roles:[],errors:[]}:undefined}); await f.flush(); assert.equal(f.get('role-load-models').disabled,false);
  removed=true; await f.click('refresh-roles'); assert.equal(f.get('role-load-models').disabled,true); assert.equal(f.get('role-ref-text').value,'');
});

test('已完成 prepare_role 在初始化恢复只读状态，不递归回读或误查音频', async () => {
  const state=roleState(true); state.job={id:'已完成准备',kind:'prepare_role',status:'success',message:'校验完成',audio_bytes:77};
  const f=fixture({state}); await f.flush(); assert.equal(f.calls.filter(call=>call.name==='state').length,1); assert.equal(f.calls.some(call=>call.name==='audio'),false); assert.equal(f.get('role-load-models').disabled,false);
});

function fixture(options = {}) {
  const elements = new Map();
  for (const match of html.matchAll(/<([a-z][\w-]*)\b([^>]*\bid="([^"]+)"[^>]*)>/gi)) elements.set(match[3], new FakeElement(match[3], match[2]));
  for (const match of html.matchAll(/<textarea\b[^>]*id="([^"]+)"[^>]*>([\s\S]*?)<\/textarea>/g)) elements.get(match[1]).value = match[2];
  const get = (id) => { assert.ok(elements.has(id), '未知元素：' + id); return elements.get(id); };
  const document = { getElementById: get, createElement: () => new FakeElement(), documentElement: { lang: 'zh-CN' } };
  const state = options.state || defaultState(); const calls = []; const timers = new Map(); let timerSequence = 0; const events = new Map(); const blobs = []; const revoked = []; let original = null;
  class FakeURL extends URL { static createObjectURL(blob) { blobs.push(blob); return 'blob:试听-' + blobs.length; } static revokeObjectURL(url) { revoked.push(url); } }
  const sdk = { ready: options.ready || Promise.resolve(), query: async (name, args) => {
    validateArguments(declaration.pages[0].queries[name], args); calls.push({ kind: 'query', name, args: json(args) });
    if (options.query) { const result = await options.query(name, args, state); if (result !== undefined) return result; }
    if (name === 'state') return json(state);
    if (name === 'ensure_connection') { state.connection = {status:'success',message:'已自动确认保存的接口连接',checked_url:state.config.api.base_url.replace(/\/+$/, '')}; return {success:true,message:state.connection.message,connection:json(state.connection)}; }
    if (name === 'check_config') { const role = state.config.role || {}; return {success:true,ready:Boolean(state.config.voice.ref_audio_path),message:state.config.voice.ref_audio_path ? '配置已保存，可继续测试接口与试听。' : '当前配置缺少参考音频。',issues:state.config.voice.ref_audio_path ? [] : [{level:'error',field:'voice.ref_audio_path',message:'请填写参考音频路径。'}],summary:{mode:role.mode || 'original',base_model:state.config.models.base_model,role_id:role.role_id || '',emotion:role.emotion || '',emotions:role.emotions || [],api_url:state.config.api.base_url}}; }
    if (name === 'llm_tasks') return {success:true,tasks:['utils','fast_model','replyer'],current:(state.config.role || {}).llm_task || 'fast_model'};
    if (name === 'catalog') return { base_models: [{ id: 'v2', label: '第二代底模', gpt_path: 'C:/本地/预设.ckpt', sovits_path: 'C:/本地/预设.pth' }], gpt_weights: [{ label: '语义权重', path: 'C:/本地/语义.ckpt' }, {label:'GPT_weights_v2',path:'D:/训练/会见黍.ckpt'}, {label:'本机模型目录',path:'C:/本地/会见黍.ckpt'}], sovits_weights: [{ label: '声学权重', path: 'C:/本地/声学.pth' }, { label: '底模目录 · 预设', path: 'C:/本地/预设.pth', version: 'v2', mode: 'full' }, { label: '训练权重', path: 'C:/本地/训练-v4.pth', version: 'v4', mode: 'full' }], lora_weights: [{label:'角色 LoRA', path:'C:/本地/角色-lora.pth', version:'v3'}], note: '扫描完成', reference_audios: [{label:'maibot_refs', path:'C:/GPT-SoVITS/maibot_refs/平静.wav'}, {label:'maibot_refs/开心', path:'C:/GPT-SoVITS/maibot_refs/开心/开心.wav'}], reference_dir: 'C:/GPT-SoVITS/maibot_refs', reference_note: '在 C:/GPT-SoVITS/maibot_refs 中找到 2 个参考音频。' };
    if (name === 'roles') return {data_dir:'C:/GPT-SoVITS/maibot_roles',scanned_paths:['C:/GPT-SoVITS/maibot_roles'],roles:[{id:'黍',name:'黍',emotions:['happy','angry','surprised','neutral','sad'].map((id) => ({id,label:'情绪：' + id})),json_path:'data/黍/黍.json'},{id:'另一个角色',name:'另一个角色',emotions:[{id:'neutral',label:'平静'}],json_path:'data/另一个角色/角色.json'}],errors:[]};
    if (name === 'job') {
      state.job = {...json(state.job),status:'success',message:'任务完成'};
      if (state.job.kind === 'prepare_role') applyRoleState(state);
      if (state.job.kind === 'load_models') state.model_status = {status:'success',message:'加载成功'};
      return json(state.job);
    }
    throw new Error('未配置查询：' + name);
  }, action: async (name, args) => {
    validateArguments(declaration.pages[0].actions[name], args); calls.push({ kind: 'action', name, args: json(args) });
    if (options.action) { const result = await options.action(name, args, state); if (result !== undefined) return result; }
    if (name === 'set_mode') {
      if (args.mode === 'role' && state.config.role.mode !== 'role') original = json({models:state.config.models,voice:state.config.voice});
      if (args.mode === 'original' && original) { state.config.models = original.models; state.config.voice = original.voice; }
      state.config.role.mode = args.mode; state.model_status = {status:'idle',message:'请手动加载'};
    }
    if (name === 'prepare_role') { state.config.role = {...state.config.role,mode:'role',...json(args)}; state.job = {id:'prepare-' + calls.length,kind:'prepare_role',status:'running',message:'正在校验角色资源'}; return {success:true,message:'已启动',job:json(state.job)}; }
    if (name === 'save_base') { const base_model = args.sovits_weights === 'C:/本地/预设.pth' ? 'v2' : 'custom'; Object.assign(state.config.models,json(args),{base_model,lora_weights:''}); if (base_model === 'v2') state.config.models.gpt_weights = 'C:/本地/预设.ckpt'; state.role_status={status:'idle',message:'底模已更新，请重新读取角色配置',profile:null}; state.model_status={status:'idle',message:'底模已保存，请手动加载'}; return {success:true,base_model,message:base_model === 'v2' ? '已保存底模：v2（自动识别）' : '已保存 SoVITS 权重，按自定义权重处理'}; }
    if (name === 'save_auto_voice') state.config.auto_voice = json(args);
    if (name === 'save_emotion_judge') { Object.assign(state.config.role, json(args)); return {success:true,message:args.llm_emotion ? '已保存：多情感时使用 MaiBot 的 ' + args.llm_task + ' 判断每句话的情感' : '已关闭 LLM 情感判断'}; }
    if (name === 'test_emotion') return {success:true,emotion:'happy',label:'开心',model:'gpt-fast',message:'判断结果：开心（happy） · 模型：gpt-fast'};
    if (name === 'save_parameters') { for (const key of parameterKeys) state.config[['max_text_length','cooldown_seconds','send_text'].includes(key) ? 'behavior' : 'voice'][key] = args[key]; }
    if (name === 'save_api') Object.assign(state.config.api, json(args));
    if (name === 'test_api') state.connection = { status: 'success', message: '连接成功' };
    if (name === 'save_models') { for (const key of ['root', 'base_model', 'gpt_weights', 'sovits_weights', 'lora_weights']) state.config.models[key] = args[key]; for (const key of ['ref_audio_path', 'prompt_text', 'prompt_lang', 'text_lang']) state.config.voice[key] = args[key]; }
    if (name === 'start_job') { state.job = { id: 'job-' + calls.length, kind: args.kind, status: 'running', message: '正在执行', audio_bytes: 0, duration_seconds: 0 }; return { success: true, message: '已启动', job: json(state.job) }; }
    if (name === 'cancel_job') state.job.status = 'cancelled';
    return { success: true, message: '操作成功' };
  } };
  const requests = []; const sdkConcurrency = {active:0,max:0,violations:0};
  for (const kind of ['query','action']) {
    const dispatch = sdk[kind];
    sdk[kind] = async (name,args) => {
      sdkConcurrency.active += 1; sdkConcurrency.max = Math.max(sdkConcurrency.max,sdkConcurrency.active);
      const request = {kind,name,args:json(args)}; requests.push(request);
      try {
        if (sdkConcurrency.active > 1) { sdkConcurrency.violations += 1; throw new Error('Two requests are already running'); }
        if (options.sdkGate) await options.sdkGate(request);
        return await dispatch(name,args);
      } finally { sdkConcurrency.active -= 1; }
    };
  }
  const window = { addEventListener(name, listener) { const items = events.get(name) || []; items.push(listener); events.set(name, items); } }; if (!options.noSDK) window.maibot = sdk;
  const context = vm.createContext({ window, document, URL: FakeURL, Blob, Uint8Array, atob, btoa, console, setTimeout(callback, ms) { const id = ++timerSequence; timers.set(id, { callback, ms }); return id; }, clearTimeout(id) { timers.delete(id); } });
  new vm.Script(source, { filename: 'webui/setup.html' }).runInContext(context);
  const flush = async () => { for (let i = 0; i < 80; i += 1) await Promise.resolve(); };
  return { get, state, calls, requests, sdkConcurrency, timers, blobs, revoked, flush, events,
    async click(id) { assert.equal(get(id).disabled, false, id + ' 不应被禁用'); get(id).emit('click'); await flush(); },
    async checkEmotion(id, checked) { const checkbox = get('emotion-options').children.map(item=>item.children[0]).find(item=>item.value===id); assert.ok(checkbox,'缺少情感复选框：' + id); assert.equal(checkbox.disabled,false); checkbox.checked=checked; checkbox.emit('change'); await flush(); },
    async input(id, value, event = 'input') { get(id).value = value; get(id).emit(event); await flush(); },
    async tick(ms = 1000) { const entry = [...timers.entries()].find(([, timer]) => timer.ms === ms); assert.ok(entry, '找不到预期计时器：' + ms); timers.delete(entry[0]); entry[1].callback(); await flush(); },
    dispose() { for (const listener of events.get('pagehide') || []) listener(); }
  };
}

test('无 SDK 明确提示 MaiBot 内打开，不伪造成功', async () => {
  const f = fixture({ noSDK: true }); await f.flush();
  assert.match(f.get('global-message').textContent, /请在 MaiBot 内打开/); assert.equal(f.get('test-api').disabled, true); assert.equal(f.calls.length, 0);
});

test('初始读取前禁用，保存接口先于测试，编辑立即清除已验证状态', async () => {
  let release; const f = fixture({ ready: new Promise((resolve) => { release = resolve; }) });
  assert.equal(f.get('base-url').disabled, true); assert.equal(f.calls.length, 0); release(); await f.flush();
  assert.equal(f.get('base-url').value, 'http://127.0.0.1:9880'); assert.equal(f.get('next-step').disabled, true);
  await f.click('test-api'); assert.deepEqual(f.calls.filter((call) => call.kind === 'action').map((call) => call.name), ['save_api', 'test_api']);
  assert.equal(f.get('next-step').disabled, false);
  await f.input('base-url', 'http://远程服务器:9880'); assert.equal(f.get('next-step').disabled, true); assert.match(f.get('api-dirty').textContent, /未保存/);
});

test('连接失败时不能进入后续步骤，错误按钮可恢复，成功后才解锁', async () => {
  let fail = true;
  const f = fixture({ action: (name) => name === 'test_api' && fail ? { success: false, message: '连接失败，请检查地址' } : undefined }); await f.flush();
  await f.click('test-api'); assert.match(f.get('global-message').textContent, /操作失败/); assert.equal(f.get('test-api').disabled, false);
  assert.equal(f.get('operation-error').hidden, false); assert.match(f.get('operation-error-message').textContent, /第1步 · API 配置测试：连接失败，请检查地址/);
  assert.equal(f.get('status-step-1').dataset.tone, 'error'); assert.match(f.get('status-step-1-text').textContent, /连接失败/);
  for (const id of ['step-base', 'step-models', 'step-parameters', 'step-auto', 'next-step']) assert.equal(f.get(id).disabled, true, id);
  assert.match(f.get('step-models').title, /请先完成第1步「API 配置测试」/);
  await f.click('status-step-3'); assert.equal(currentStep(f), 1); assert.match(f.get('global-message').textContent, /请先完成第1步「API 配置测试」：接口测试失败：连接失败，请检查地址/);
  assert.equal(f.get('load-models').disabled, true);
  fail = false; await f.click('test-api'); assert.equal(f.get('step-base').disabled, false); assert.equal(f.get('next-step').disabled, false);
  await f.click('next-step'); assert.equal(currentStep(f), 2);
});

test('第二步按所选 SoVITS 权重自动识别底模版本，无需用户选择', async () => {
  const f = fixture(); await f.flush();
  assert.equal(f.get('base-model').readOnly, true);
  assert.equal(f.get('base-model').value, '自定义权重（未识别为已安装的标准底模）'); assert.match(f.get('base-model-help').textContent, /扫描结果中没有该文件/);
  await f.input('sovits-model-picker', 'C:/本地/预设.pth', 'change'); assert.equal(f.get('base-model').value, 'v2'); assert.match(f.get('base-model-help').textContent, /自动识别为 v2/);
  await f.input('sovits-model-picker', 'C:/本地/训练-v4.pth', 'change'); assert.match(f.get('base-model').value, /自定义权重/); assert.match(f.get('base-model-help').textContent, /未能识别/);
  await f.input('sovits-model-picker', '', 'change'); assert.match(f.get('base-model').value, /请先选择/);
  await f.input('sovits-model-picker', 'C:/本地/预设.pth', 'change'); assert.match(f.get('base-dirty').textContent, /未保存/);
  const path = 'C:/本地/语义.ckpt'; await f.input('gpt-model-picker', path, 'change'); await f.click('save-models');
  const saved = f.calls.findLast((call) => call.name === 'save_models').args;
  assert.equal(saved.gpt_weights, path); assert.equal(saved.base_model, 'v2'); assert.equal(saved.sovits_weights, 'C:/本地/预设.pth');
  assert.equal(f.calls.some((call) => call.name === 'start_job'), false);
});
test('加载先保存、单计时器轮询，声音参数编辑不清模型就绪，权重编辑会清除', async () => {
  const f = fixture(); await f.flush(); await f.click('test-api'); await f.click('step-models');
  await f.click('load-models');
  assert.deepEqual(f.calls.filter((call) => call.kind === 'action').slice(-2).map((call) => call.name), ['save_models', 'start_job']);
  assert.equal(f.calls.findLast((call) => call.name === 'start_job').args.kind, 'load_models');
  assert.equal([...f.timers.values()].filter((timer) => timer.ms === 1000).length, 1);
  await f.tick(); assert.equal(f.get('synthesize').disabled, false); assert.match(f.get('models-label').textContent, /就绪/);
  await f.input('ref-audio-path', 'C:/GPT-SoVITS/maibot_refs/开心/开心.wav', 'change'); await f.input('prompt-text', '你好'); assert.equal(f.get('synthesize').disabled, false);
  await f.input('gpt-model-picker', 'C:/本地/语义.ckpt', 'change'); assert.equal(f.get('synthesize').disabled, true); assert.match(f.get('models-dirty').textContent, /未保存/);
});

test('加载过程中编辑权重不被任务结果误判就绪，刷新不覆写输入', async () => {
  const f = fixture(); await f.flush(); await f.click('test-api'); await f.click('load-models');
  await f.input('gpt-model-picker', 'C:/本地/语义.ckpt', 'change'); await f.tick(); assert.equal(f.get('synthesize').disabled, true); assert.match(f.get('model-message').textContent, /已经修改/);
  await f.click('refresh-state'); assert.equal(f.get('gpt-model-picker').value, 'C:/本地/语义.ckpt'); assert.equal(f.get('synthesize').disabled, true);
});

test('刷新恢复后台任务，卸载仅清理计时器不取消用户任务', async () => {
  const state = defaultState(); state.connection.status = 'success'; state.job = { id: '已有任务', kind: 'load_models', status: 'running', message: '正在加载', audio_bytes: 0, duration_seconds: 0 };
  const f = fixture({ state }); await f.flush(); assert.equal([...f.timers.values()].filter((timer) => timer.ms === 1000).length, 1);
  await f.click('refresh-state'); assert.equal([...f.timers.values()].filter((timer) => timer.ms === 1000).length, 1);
  f.dispose(); assert.equal(f.timers.size, 0); assert.equal(f.calls.some((call) => call.name === 'cancel_job'), false);
});

function audioFixture(invalidOffset = false) {
  const state = defaultState(); state.connection.status = 'success'; state.model_status.status = 'success'; state.config.voice.ref_audio_path = 'C:/参考.wav';
  const bytes = Buffer.alloc(262151, 7); bytes.write('RIFF', 0); bytes.write('WAVE', 8);
  const query = (name, args, saved) => {
    if (name === 'job') return { ...json(saved.job), status: 'success', audio_bytes: bytes.length, duration_seconds: 1.25 };
    if (name === 'audio') { const next = Math.min(args.offset + 131072, bytes.length); return { base64: bytes.subarray(args.offset, next).toString('base64'), next_offset: invalidOffset ? args.offset : next, done: next === bytes.length, total_bytes: bytes.length }; }
  };
  return { ...fixture({ state, query }), bytes };
}

test('合成先保存不自动加载，独立解码每块且原始字节偏移正确，可下载不自动播放', async () => {
  const f = audioFixture(); await f.flush(); await f.click('step-models'); await f.click('synthesize');
  assert.deepEqual(f.calls.filter((call) => call.kind === 'action').map((call) => call.name), ['save_models', 'start_job']);
  assert.equal(f.calls.findLast((call) => call.name === 'start_job').args.kind, 'synthesize'); await f.tick();
  assert.deepEqual(f.calls.filter((call) => call.name === 'audio').map((call) => call.args.offset), [0, 131072, 262144]);
  assert.equal(f.blobs.length, 1); assert.deepEqual(Buffer.from(await f.blobs[0].arrayBuffer()), f.bytes);
  assert.equal(f.get('audio-block').hidden, false); assert.equal(f.get('download-audio').href, 'blob:试听-1');
  assert.match(f.get('global-message').textContent, /手动点击/); await f.input('prompt-text', '修改参考文字'); assert.deepEqual(f.revoked, ['blob:试听-1']); assert.equal(f.get('audio-block').hidden, true);
});

test('音频偏移不前进立即失败，不死循环、不创建 Blob URL', async () => {
  const f = audioFixture(true); await f.flush(); await f.click('synthesize'); await f.tick();
  assert.equal(f.calls.filter((call) => call.name === 'audio').length, 1); assert.equal(f.blobs.length, 0); assert.match(f.get('global-message').textContent, /音频偏移量无效/); assert.equal(f.get('retry-audio').hidden, false);
});

test('只有成功的合成任务且有音频才读取 audio 查询', async () => {
  for (const [kind, status, audio_bytes] of [['load_models', 'success', 99], ['synthesize', 'error', 99], ['synthesize', 'cancelled', 99], ['synthesize', 'success', 0]]) {
    const state = defaultState(); state.job = { id: '结束任务', kind, status, audio_bytes, duration_seconds: 0, message: '任务结束' }; const f = fixture({ state }); await f.flush();
    assert.equal(f.calls.some((call) => call.name === 'audio'), false); assert.equal(f.blobs.length, 0);
  }
});

test('轮询错误可继续查询且不会产生并行计时器', async () => {
  let fail = true; const f = fixture({ query: (name) => { if (name === 'job' && fail) throw new Error('连接暂时中断'); } });
  await f.flush(); await f.click('test-api'); await f.click('load-models'); await f.tick();
  assert.equal(f.get('retry-job').hidden, false); assert.equal([...f.timers.values()].filter((timer) => timer.ms === 1000).length, 0);
  fail = false; await f.click('retry-job'); assert.equal([...f.timers.values()].filter((timer) => timer.ms === 1000).length, 1); await f.tick(); assert.equal(f.get('synthesize').disabled, false);
});

test('GPT 和 SoVITS 下拉选择写入完整路径并清除就绪，不自动加载', async () => {
  const state = defaultState(); state.connection.status = 'success'; state.model_status.status = 'success';
  const f = fixture({state}); await f.flush(); assert.equal(f.get('synthesize').disabled, false);
  await f.input('gpt-model-picker', 'D:/训练/会见黍.ckpt', 'change');
  assert.equal(f.get('gpt-path').textContent, '完整路径：D:/训练/会见黍.ckpt');
  assert.equal(f.get('synthesize').disabled, true); assert.match(f.get('models-dirty').textContent, /未保存/);
  await f.input('sovits-model-picker', 'C:/本地/声学.pth', 'change');
  assert.match(f.get('base-dirty').textContent, /未保存/); assert.match(f.get('base-model').value, /自定义权重/);
  assert.equal(f.calls.some((call) => call.kind === 'action'), false);
  const duplicates = f.get('gpt-model-picker').options.filter((option) => option.value.endsWith('/会见黍.ckpt'));
  assert.equal(duplicates.length, 2); assert.notEqual(duplicates[0].textContent, duplicates[1].textContent);
  for (const option of duplicates) { assert.ok(option.textContent.includes('会见黍.ckpt')); assert.ok(option.textContent.includes(option.value)); assert.equal(option.title, option.value); }
});

test('SoVITS 下拉显示文件名、来源与完整路径，已保存但未扫描到的路径保留可见', async () => {
  const f = fixture(); await f.flush();
  const options = f.get('sovits-model-picker').options;
  assert.equal(options[0].value, ''); assert.equal(f.get('sovits-model-picker').value, 'C:/模型/声学.pth');
  const saved = options.find((option) => option.value === 'C:/模型/声学.pth'); assert.ok(saved); assert.match(saved.textContent, /^已保存：声学\.pth · C:\/模型\/声学\.pth$/);
  const preset = options.find((option) => option.value === 'C:/本地/预设.pth');
  assert.equal(preset.textContent, '预设.pth · v2 · 底模目录 · C:/本地/预设.pth'); assert.equal(preset.title, 'C:/本地/预设.pth');
  assert.equal(options.find((option) => option.value === 'C:/本地/声学.pth').textContent, '声学.pth · 声学权重 · C:/本地/声学.pth');
  assert.equal(f.get('sovits-path').textContent, '完整路径：C:/模型/声学.pth');
  await f.input('sovits-model-picker', '', 'change'); assert.equal(f.get('sovits-path').textContent, '尚未选择 SoVITS 权重。');
  assert.equal(f.calls.some((call) => call.name === 'start_job'), false);
});
test('GPT 与 LoRA 只通过下拉选择，已保存但未扫描到的路径保留为已保存选项并显示完整路径', async () => {
  const state = defaultState(); state.config.models.lora_weights = 'D:/远程/未收录-lora.pth';
  const f = fixture({state}); await f.flush();
  const gpt = f.get('gpt-model-picker'); assert.equal(gpt.value, 'C:/模型/语义.ckpt');
  assert.equal(gpt.options.find((option) => option.value === 'C:/模型/语义.ckpt').textContent, '已保存：语义.ckpt · C:/模型/语义.ckpt');
  assert.equal(f.get('gpt-path').textContent, '完整路径：C:/模型/语义.ckpt');
  assert.equal(f.get('lora-model-picker').value, 'D:/远程/未收录-lora.pth'); assert.equal(f.get('lora-path').textContent, '完整路径：D:/远程/未收录-lora.pth');
  await f.input('lora-model-picker', '', 'change'); assert.equal(f.get('lora-path').textContent, '不使用 LoRA，只加载底模。');
  await f.input('gpt-model-picker', 'C:/本地/语义.ckpt', 'change'); assert.equal(f.get('gpt-path').textContent, '完整路径：C:/本地/语义.ckpt');
  assert.equal(gpt.options.find((option) => option.value === 'C:/本地/语义.ckpt').textContent, '语义.ckpt · 语义权重 · C:/本地/语义.ckpt');
  assert.equal(f.calls.some((call) => call.kind === 'action'), false);
});
test('重新扫描与刷新状态保留手填路径，扫描不会伪造模型重新就绪', async () => {
  const state = defaultState(); state.connection.status = 'success'; state.model_status.status = 'success';
  const f = fixture({state}); await f.flush();
  await f.input('gpt-model-picker', 'C:/本地/会见黍.ckpt', 'change'); await f.input('sovits-model-picker', 'C:/本地/预设.pth', 'change');
  await f.click('scan-models'); await f.click('refresh-state');
  assert.equal(f.get('gpt-model-picker').value, 'C:/本地/会见黍.ckpt'); assert.equal(f.get('sovits-model-picker').value, 'C:/本地/预设.pth'); assert.equal(f.get('base-model').value, 'v2');
  assert.equal(f.get('synthesize').disabled, true); assert.equal(f.calls.some((call) => call.name === 'start_job'), false);
});

test('只保留一个本机模型目录：扫描只传 root，目录过长不发请求且原因显示在左栏底部', async () => {
  const state = defaultState(); state.connection.status = 'success'; state.model_status.status = 'success';
  const f = fixture({state}); await f.flush();
  assert.deepEqual(f.calls.find((call) => call.name === 'catalog').args, {root:'C:/GPT-SoVITS'});
  assert.ok(html.includes('<label for="model-root">本机模型目录</label>')); assert.doesNotMatch(html, /本机模型目录（可选）|补充模型目录/);
  await f.input('model-root', 'D:/主目录'); assert.equal(f.get('synthesize').disabled, false); assert.match(f.get('base-dirty').textContent, /未保存/);
  await f.click('save-models'); const saved = f.calls.findLast((call) => call.name === 'save_models').args;
  assert.equal(saved.root, 'D:/主目录');
  const sent = () => f.calls.filter((call) => call.name !== 'check_config').length;
  await f.input('model-root', '长'.repeat(1025)); const count = sent(); await f.click('scan-models');
  assert.equal(sent(), count); assert.equal(f.get('operation-error').hidden, false); assert.match(f.get('operation-error-message').textContent, /第2步 · 选择底模：本机模型目录超过允许长度/); assert.match(f.get('global-message').textContent, /左栏最下方/);
  await f.click('save-models'); assert.equal(sent(), count);
});
test('扫描过程中任意目录变化均忽略旧结果', async () => {
  for (const field of ['model-root']) {
    let release; let deferred = false;
    const f = fixture({query: (name) => name === 'catalog' && deferred ? new Promise((resolve) => { release = resolve; }) : undefined}); await f.flush();
    deferred = true; await f.click('scan-models'); await f.input(field, 'D:/已更改');
    release({base_models:[],gpt_weights:[{label:'过期',path:'D:/旧结果.ckpt'}],sovits_weights:[],note:'不应应用'}); await f.flush();
    assert.match(f.get('catalog-note').textContent, /旧扫描结果已忽略/);
    assert.equal(f.get('gpt-model-picker').options.some((option) => option.value === 'D:/旧结果.ckpt'), false);
  }
});


test('真实配置侧栏没有品牌装饰，并说明目录入口及独立 LoRA 语义', () => {
  assert.doesNotMatch(html, /brand-mark|hero/i);
  assert.match(html, /<aside class="side-column"/); assert.match(html, /id="check-config-card"/);
  assert.match(html, /\.layout\{display:grid;grid-template-columns:minmax\(0,3fr\) minmax\(300px,2fr\)/);
  assert.match(html, /@media\(max-width:899px\)\{\.layout\{grid-template-columns:minmax\(0,1fr\)/);
  assert.doesNotMatch(html, /--local-/);
  assert.match(html, /没有自定义底模 API/); assert.match(html, /不能任意把不匹配的底模与 LoRA 配对/);
  assert.match(html, /label for="lora-model-picker"/); assert.match(html, /label for="gpt-model-picker"/);
});

test('LoRA 可空、空目录明确提示且不把 GPT ckpt 当 LoRA', async () => {
  const f = fixture({query:(name) => name === 'catalog' ? {base_models:[],gpt_weights:[{label:'GPT',path:'D:/GPT.ckpt'}],sovits_weights:[],lora_weights:[]} : undefined});
  await f.flush();
  assert.equal(f.get('lora-model-picker').value, ''); assert.equal(f.get('lora-path').textContent, '不使用 LoRA，只加载底模。');
  assert.match(f.get('lora-model-picker').options[0].textContent, /无LoRA/);
  assert.match(f.get('lora-catalog-note').textContent, /本机模型目录中未发现 LoRA/);
  assert.equal(f.get('lora-model-picker').options.length, 1);
  await f.click('save-models'); assert.equal(f.calls.findLast(call => call.name === 'save_models').args.lora_weights, '');
  assert.equal(f.calls.some(call => call.name === 'start_job'), false);
});

test('LoRA 选择清除就绪且保存不加载，空 LoRA 保留原有加载接口', async () => {
  const state = defaultState(); state.connection.status = 'success'; state.model_status.status = 'success';
  const f = fixture({state}); await f.flush(); assert.equal(f.get('synthesize').disabled, false);
  const gpt = f.get('gpt-model-picker').value; const base = f.get('sovits-model-picker').value;
  await f.input('lora-model-picker', 'C:/本地/角色-lora.pth', 'change');
  assert.equal(f.get('lora-path').textContent, '完整路径：C:/本地/角色-lora.pth'); assert.equal(f.get('synthesize').disabled, true);
  assert.equal(f.get('gpt-model-picker').value, gpt); assert.equal(f.get('sovits-model-picker').value, base);
  assert.equal(f.calls.some(call => call.kind === 'action'), false);
  await f.click('save-models'); assert.equal(f.calls.findLast(call => call.name === 'save_models').args.lora_weights, 'C:/本地/角色-lora.pth');
  assert.equal(f.calls.some(call => call.name === 'start_job'), false);
  await f.input('lora-model-picker', '', 'change'); assert.equal(f.get('lora-model-picker').value, '');
  await f.click('load-models');
  assert.deepEqual(f.calls.findLast(call => call.name === 'start_job').args, {kind:'load_models',text:''});
  assert.equal(f.calls.findLast(call => call.name === 'save_models').args.lora_weights, '');
  assert.equal(f.calls.findLast(call => call.name === 'save_models').args.sovits_weights, base);
});

test('LoRA 重扫与刷新保留当前选择，签名必须匹配 LoRA', async () => {
  const state = defaultState(); state.connection.status = 'success'; state.model_status.status = 'success';
  const f = fixture({state}); await f.flush(); assert.equal(f.get('synthesize').disabled, false);
  await f.input('lora-model-picker', 'C:/本地/角色-lora.pth', 'change'); assert.equal(f.get('synthesize').disabled, true);
  await f.click('scan-models'); await f.click('refresh-state');
  assert.equal(f.get('lora-model-picker').value, 'C:/本地/角色-lora.pth'); assert.equal(f.get('synthesize').disabled, true);
  assert.equal(f.calls.some(call => call.name === 'start_job'), false);
});
test('已加载 LoRA 状态读取及原生空值兼容，目录不进入加载签名', async () => {
  for (const saved of ['C:/本地/角色-lora.pth', '（不使用 LoRA）']) {
    const state = defaultState(); state.config.models.lora_weights = saved; state.connection.status = 'success'; state.model_status.status = 'success';
    const f = fixture({state}); await f.flush();
    assert.equal(f.get('lora-model-picker').value, saved === '（不使用 LoRA）' ? '' : saved); assert.equal(f.get('synthesize').disabled, false);
    await f.input('model-root', 'D:/仅目录变化'); assert.equal(f.get('synthesize').disabled, false);
    await f.input('lora-model-picker', saved === '（不使用 LoRA）' ? 'C:/本地/角色-lora.pth' : '', 'change'); assert.equal(f.get('synthesize').disabled, true);
  }
});

test('状态初始化失败仍可查看编辑及刷新恢复，不永久禁用表单', async () => {
  let fail = true;
  const f = fixture({query:(name) => { if (name === 'state' && fail) throw new Error('配置读取失败'); }}); await f.flush();
  assert.match(f.get('global-message').textContent, /仍可查看和编辑表单/);
  assert.equal(f.get('base-url').disabled, false); assert.equal(f.get('refresh-state').disabled, false);
  // 状态未读取时第一步未完成，不能跳到后续步骤；表单本身仍可编辑。
  assert.equal(f.get('step-models').disabled, true); assert.equal(f.get('lora-model-picker').disabled, false);
  assert.equal(f.get('load-models').disabled, true);
  fail = false; await f.click('refresh-state'); assert.equal(f.get('base-url').value, 'http://127.0.0.1:9880');
  assert.ok(f.get('lora-model-picker').options.some(o => o.value === 'C:/本地/角色-lora.pth'));
});


test('插件禁用仍可读取和编辑模型，加载与试听有明确中文提示', async () => {
  const state = defaultState(); state.config.plugin.enabled = false; state.connection.status = 'success'; state.model_status.status = 'success';
  const f = fixture({state}); await f.flush(); await f.click('step-models');
  assert.equal(f.get('lora-model-picker').disabled, false); assert.equal(f.get('gpt-model-picker').disabled, false);
  assert.equal(f.get('load-models').disabled, true); assert.equal(f.get('synthesize').disabled, true);
  assert.match(f.get('connection-required').textContent, /插件当前已禁用/); assert.match(f.get('synthesize-requirement').textContent, /启用/);
});

test('LoRA 目录忽略 ckpt，已保存配置里的 GPT ckpt 不可保存为 LoRA', async () => {
  const state = defaultState(); state.config.models.lora_weights = 'D:/GPT.ckpt';
  const f = fixture({state, query:(name) => name === 'catalog' ? {base_models:[],gpt_weights:[],sovits_weights:[],lora_weights:[{label:'非LoRA',path:'D:/其他.ckpt'}]} : undefined});
  await f.flush(); assert.equal(f.get('lora-model-picker').options.some(o => o.value === 'D:/其他.ckpt'), false);
  assert.match(f.get('lora-catalog-note').textContent, /未发现 LoRA/); assert.equal(f.get('lora-model-picker').value, 'D:/GPT.ckpt');
  assert.equal(f.get('status-step-3').dataset.tone, 'error'); assert.match(f.get('status-step-3-text').textContent, /LoRA 必须是 \.pth/);
  await f.click('save-models');
  assert.match(f.get('operation-error-message').textContent, /第3步 · 声音模式：.*不能使用 GPT/); assert.equal(f.calls.some(call => call.kind === 'action'), false);
});
test('五步真实结构：底模第二步、模式第三步（无上传）、参数与可选试听第四步、自动语音第五步', () => {
  const section = id => html.match(new RegExp('<section id="' + id + '"[\\s\\S]*?<\\/section>'))?.[0];
  const base = section('base-card'), original = section('panel-models'), role = section('role-card');
  assert.ok(base); assert.ok(original); assert.ok(role);
  for (const id of ['model-root','sovits-model-picker','base-model','scan-models','save-base']) assert.match(base,new RegExp('id="' + id + '"'));
  // 没有跳过未完成步骤的入口。
  assert.doesNotMatch(html, /id="next-mode"|id="inspect-models"|不保存，浏览第三步|暂不测试连接/);
  for (const id of ['gpt-model-picker','lora-model-picker','ref-audio-path','role-picker','model-extra-dir','sovits-weights']) assert.doesNotMatch(base,new RegExp('id="' + id + '"'));
  // 参考音频在第三步普通模式中用下拉框选择，第四步只剩生成参数。
  for (const id of ['ref-audio-path','refresh-refs','ref-audio-full','refs-note','prompt-text','prompt-lang','text-lang']) assert.match(original,new RegExp('id="' + id + '"'), id);
  assert.match(original,/<select id="ref-audio-path" class="maibot-select path"/); assert.doesNotMatch(html,/<input id="ref-audio-path"/);
  assert.doesNotMatch(original,/id="(?:base-model|model-root|sovits-model-picker|scan-models)"/);
  assert.doesNotMatch(original,/id="(?:reference-advanced|test-card)"/);
  const parameters = section('parameters-card'); assert.doesNotMatch(parameters,/id="(?:reference-advanced|original-reference|ref-audio-path|prompt-text|role-reference)"/);
  assert.match(role,/id="role-reference"/);
  assert.match(parameters,/id="parameters-form"/); assert.match(parameters,/id="save-parameters"/);
  assert.match(html,/<div id="preview-card"[\s\S]*id="test-card"[\s\S]*id="job-card"[\s\S]*?<\/main>/);
  assert.ok(html.indexOf('id="step-models"') < html.indexOf('id="step-parameters"')); assert.ok(html.indexOf('id="step-parameters"') < html.indexOf('id="step-auto"'));
  assert.doesNotMatch(html, /id="step-preview"|id="next-preview"|id="preview-back"|id="status-step-6"/); assert.equal([...html.matchAll(/class="step maibot-button"/g)].length, 5);
  assert.match(html, /<span>4 · 参数与试听<\/span>/); assert.match(html, /<span>5 · 自动语音回复<\/span>/); assert.match(html, /第四步 · 试听（可选）/); assert.match(html, /不一定要试听：保存参数后可直接进入第五步/);
  assert.match(original,/id="gpt-model-picker"/); assert.match(original,/id="lora-model-picker"/); assert.doesNotMatch(original,/id="gpt-weights"|id="lora-weights"/);
  assert.match(section('mode-card'),/id="original-mode-btn"/); assert.match(section('mode-card'),/id="role-mode-btn"/);
  assert.doesNotMatch(role,/id="upload-|type="file"/); assert.match(role,/id="emotion-options"/);
  assert.ok(html.indexOf('id="step-api"') < html.indexOf('id="step-base"')); assert.ok(html.indexOf('id="step-base"') < html.indexOf('id="step-models"'));
});

test('插件禁用且未连接时不能浏览后续步骤；已到达的步骤可随时返回', async () => {
  const state=defaultState(); state.config.plugin.enabled=false; const f=fixture({state}); await f.flush();
  assert.equal(f.get('mode-card').hidden,true); assert.equal(f.get('base-card').hidden,true);
  for (const id of ['step-base','step-models','next-step']) assert.equal(f.get(id).disabled,true,id);
  const g=fixture({state:connectedState()}); await g.flush();
  await g.click('next-step'); assert.equal(g.get('base-card').hidden,false); assert.equal(g.get('step-base').attrs.get('aria-current'),'step');
  await g.click('step-models'); assert.equal(g.get('mode-card').hidden,false); assert.equal(g.get('panel-models').hidden,false); assert.equal(g.get('step-models').attrs.get('aria-current'),'step');
  await g.click('back-step'); assert.equal(g.get('base-card').hidden,false); await g.click('base-back-step'); assert.equal(g.get('panel-api').hidden,false);
  await g.click('step-models'); await g.click('role-mode-btn'); await g.click('role-back-step'); assert.equal(g.get('base-card').hidden,false); assert.equal(g.get('role-card').hidden,true);
});

test('API 下一步进入底模，save_base 只发目录与 SoVITS 权重并按识别结果继续第三步', async () => {
  const f=fixture(); await f.flush(); await f.click('test-api'); await f.click('next-step'); assert.equal(f.get('base-card').hidden,false);
  await f.input('lora-model-picker','C:/本地/角色-lora.pth','change'); await f.input('sovits-model-picker','C:/本地/预设.pth','change'); assert.equal(f.get('base-model').value,'v2'); await f.click('save-base');
  assert.deepEqual(f.calls.findLast(call=>call.name==='save_base').args,{root:'C:/GPT-SoVITS',sovits_weights:'C:/本地/预设.pth'});
  assert.equal(f.calls.filter(call=>call.name==='save_models'||call.name==='start_job').length,0); assert.equal(f.get('mode-card').hidden,false); assert.equal(f.get('base-card').hidden,true);
  assert.equal(f.get('lora-model-picker').value,''); assert.equal(f.get('gpt-model-picker').value,'C:/本地/预设.ckpt'); assert.match(f.get('base-dirty').textContent,/已保存/); assert.equal(f.get('base-model').value,'v2');
  assert.deepEqual(f.calls.filter(call=>call.kind==='query').slice(-2).map(call=>call.name),['state','check_config']); assert.match(f.get('global-message').textContent,/v2/);
  assert.equal(f.get('status-step-2').dataset.tone,'success'); assert.match(f.get('status-step-2-text').textContent,/v2 底模 · 预设\.pth/);
});
test('save_base 失败或保存期间修改不继续，也不覆盖最新底模选择', async () => {
  const fail=fixture({state:connectedState(),action:name=>name==='save_base'?{success:false,message:'底模保存失败'}:undefined}); await fail.flush(); await fail.click('step-base'); await fail.click('save-base');
  assert.equal(fail.get('base-card').hidden,false); assert.equal(fail.get('operation-error').hidden,false); assert.match(fail.get('operation-error-message').textContent,/第2步 · 选择底模：底模保存失败/); assert.match(fail.get('global-message').textContent,/操作失败/);
  await fail.click('refresh-state'); assert.equal(fail.get('operation-error').hidden,true);
  // 未选择底模时“保存底模并继续”不可点击，也不会发出保存请求。
  await fail.input('sovits-model-picker','','change'); assert.equal(fail.get('save-base').disabled,true); assert.equal(fail.get('save-base').title,'请先在下拉框中选择 SoVITS 底模或完整权重');
  assert.match(fail.get('base-dirty').textContent,/请先在上方下拉框中选择 SoVITS 底模或完整权重，选择后才能保存/); assert.equal(fail.calls.filter(call=>call.name==='save_base').length,1);
  await fail.input('sovits-model-picker','C:/本地/预设.pth','change'); assert.equal(fail.get('save-base').disabled,false); assert.equal(fail.get('save-base').title,'');
  let release; const f=fixture({state:connectedState(),action:name=>name==='save_base'?new Promise(resolve=>{release=()=>resolve({success:true});}):undefined}); await f.flush(); await f.click('step-base'); await f.click('save-base');
  await f.input('model-root','D:/新目录'); release(); await f.flush(); assert.equal(f.get('base-card').hidden,false); assert.match(f.get('operation-error-message').textContent,/保存期间.*修改/); assert.equal(f.get('model-root').value,'D:/新目录');
});

test('角色模式选底模不混传JSON字段，回读后台底模重置并撤销旧ready', async () => {
  const f=fixture({state:roleState(true)}); await f.flush(); const old=[f.get('gpt-model-picker').value,f.get('lora-model-picker').value,f.get('ref-audio-path').value];
  await f.click('step-base'); await f.input('sovits-model-picker','C:/本地/预设.pth','change');
  assert.deepEqual([f.get('gpt-model-picker').value,f.get('lora-model-picker').value,f.get('ref-audio-path').value],old); await f.click('save-base');
  assert.deepEqual(f.calls.findLast(call=>call.name==='save_base').args,{root:'C:/GPT-SoVITS',sovits_weights:'C:/本地/预设.pth'});
  assert.equal(f.get('gpt-model-picker').value,'C:/本地/预设.ckpt'); assert.equal(f.get('lora-model-picker').value,'');
  assert.equal(f.get('role-load-models').disabled,true); assert.equal(f.calls.some(call=>call.name==='save_models'),false); await f.click('refresh-state'); assert.equal(f.get('role-load-models').disabled,true);
});

test('角色目录只读固定，刷新插件data仅无参查询roles且不使已准备角色dirty', async () => {
  const f=fixture({state:roleState(true)}); await f.flush();
  assert.ok(html.includes('<label for="role-data-dir">角色目录（GPT-SoVITS 目录下的 maibot_roles）</label>'));
  assert.match(html, /<input id="role-data-dir"[^>]*\breadonly\b/); assert.equal(f.get('role-data-dir').readOnly,true);
  assert.match(html, /id="refresh-role-dir"[^>]*>刷新<\/button>/);
  assert.equal(f.get('role-data-dir').value,'C:/GPT-SoVITS/maibot_roles'); assert.equal(f.get('role-load-models').disabled,false);
  const before=f.calls.length; await f.click('refresh-role-dir');
  assert.deepEqual(f.calls.slice(before),[{kind:'query',name:'roles',args:{}},{kind:'query',name:'check_config',args:{}}]);
  assert.equal(f.get('apply-role').disabled,false); assert.equal(f.get('role-load-models').disabled,false);
  assert.match(f.get('roles-note').textContent,/C:\/GPT-SoVITS\/maibot_roles/); assert.doesNotMatch(f.get('roles-note').textContent,/plugins\/data|插件\/data/);
});

test('空角色列表显示固定own路径、JSON放置说明及错误，可刷新恢复', async () => {
  let empty=true; const own='C:/GPT-SoVITS/maibot_roles'; const f=fixture({query:name=>name==='roles'&&empty?{roles:[],errors:['角色 JSON 解析失败'],data_dir:own,scanned_paths:[own],upload_dir:own}:undefined}); await f.flush();
  for (const text of ['未找到',own,'GPT-SoVITS 安装目录下的 maibot_roles','maibot_roles/角色/role.json','JSON 解析失败','刷新角色列表']) assert.ok(f.get('roles-note').textContent.includes(text),text);
  assert.equal(f.get('role-data-dir').value,own); assert.doesNotMatch(f.get('roles-note').textContent,/自定义|绝对路径|C:\/plugins\/data/);
  assert.equal(f.get('apply-role').disabled,true); empty=false; await f.click('refresh-roles'); assert.match(f.get('roles-note').textContent,/找到 2 个角色/); assert.equal(f.get('apply-role').disabled,false);
});

test('固定角色扫描异常保留own目录诊断及刷新入口，不引导外部目录', async () => {
  let fail=false; const f=fixture({query:name=>{if(name==='roles'&&fail)throw new Error('JSON目录无法访问');}}); await f.flush();
  fail=true; await f.click('refresh-role-dir'); assert.match(f.get('roles-note').textContent,/角色扫描失败.*JSON目录无法访问/);
  assert.match(f.get('roles-note').textContent,/角色目录：C:\/GPT-SoVITS\/maibot_roles/); assert.match(f.get('roles-note').textContent,/maibot_roles\/角色\/role.json/);
  assert.equal(f.get('role-data-dir').value,'C:/GPT-SoVITS/maibot_roles'); assert.doesNotMatch(f.get('roles-note').textContent,/自定义|绝对路径/);
  fail=false; await f.click('refresh-roles'); assert.match(f.get('roles-note').textContent,/找到 2 个角色/);
});

test('角色目录只显示 roles 查询返回的路径，返回前为空', async () => {
  const state=roleState(true); let release;
  const f=fixture({state,query:name=>name==='roles'?new Promise(resolve=>{release=()=>resolve({data_dir:'C:/GPT-SoVITS/maibot_roles',scanned_paths:['C:/GPT-SoVITS/maibot_roles'],roles:[{id:'黍',emotions:[{id:'happy'}]}],errors:[]});}):undefined}); await f.flush();
  assert.ok(release); assert.equal(f.get('role-data-dir').value,''); release(); await f.flush();
  assert.equal(f.get('role-data-dir').value,'C:/GPT-SoVITS/maibot_roles'); assert.equal(f.get('role-load-models').disabled,false);
  await f.click('original-mode-btn'); await f.click('role-mode-btn');
  assert.equal(f.get('role-data-dir').value,'C:/GPT-SoVITS/maibot_roles');
});

test('两个刷新入口都只重新查询角色列表，页面声明里没有修改角色目录的接口', async () => {
  const f=fixture({state:roleState()}); await f.flush();
  await f.click('refresh-role-dir'); await f.click('refresh-roles');
  assert.equal(f.calls.filter(call=>call.kind==='action').length,0); assert.equal(f.calls.filter(call=>call.name==='roles').length,3);
  for(const call of f.calls.filter(call=>call.name==='roles'))assert.deepEqual(call.args,{});
  assert.equal(Object.values(declaration.pages[0].actions).some(action=>/roles_dir/.test(action.api)),false);
});
test('固定路径按roles结果准确显示且prepare回读state不改空或误dirty', async () => {
  const own='C:\\GPT SoVITS\\maibot_roles'; const state=roleState();
  const f=fixture({state,query:name=>name==='roles'?{data_dir:own,scanned_paths:[own],upload_dir:own,roles:[{id:'黍',emotions:[{id:'happy'}]}],errors:[]}:undefined}); await f.flush();
  assert.equal(f.get('role-data-dir').value,own); assert.ok(f.get('roles-note').textContent.includes('实际目录：'+own));
  assert.ok(f.get('roles-note').textContent.includes('扫描路径：'+own)); assert.doesNotMatch(f.get('roles-note').textContent,/上传目录/);
  await f.click('apply-role'); await f.tick(); assert.equal(f.get('role-data-dir').value,own); assert.equal(f.get('role-load-models').disabled,false);
  await f.click('refresh-state'); await f.click('refresh-role-dir'); assert.equal(f.get('role-data-dir').value,own); assert.equal(f.get('role-load-models').disabled,false);
});

test('多情感prepare携带默认加勾选集合，回读默认profile并保存全部profiles', async () => {
  const f=fixture({state:roleState()}); await f.flush(); await f.click('step-models'); await f.checkEmotion('sad',true); await f.input('emotion-picker','sad','change');
  assert.deepEqual(f.get('emotion-picker').options.map(item=>item.value),['happy','sad']); await f.click('apply-role');
  assert.deepEqual(f.calls.findLast(call=>call.name==='prepare_role').args,{role_id:'黍',emotion:'sad',emotions:['happy','sad']}); await f.tick();
  assert.equal(f.get('emotion-picker').value,'sad'); assert.equal(f.get('gpt-model-picker').value,'data/黍/sad.ckpt'); assert.equal(f.get('ref-audio-path').value,'data/黍/sad.wav');
  assert.deepEqual(Object.keys(f.state.role_status.profiles),['happy','sad']); assert.equal(f.get('role-load-models').disabled,false); assert.match(f.get('role-status').textContent,/已启用情感：happy、sad/);
  assert.equal(f.calls.some(call=>call.name==='save_models'||call.name==='start_job'),false);
});

test('情感至少一个，默认下拉仅勾选集合，取消默认后选择剩余项', async () => {
  const f=fixture({state:roleState()}); await f.flush(); await f.checkEmotion('sad',true); await f.checkEmotion('happy',false);
  assert.deepEqual(f.get('emotion-picker').options.map(item=>item.value),['sad']); assert.equal(f.get('emotion-picker').value,'sad');
  await f.checkEmotion('sad',false); assert.equal(f.get('emotion-picker').value,''); assert.equal(f.get('apply-role').disabled,true); assert.equal(f.calls.some(call=>call.name==='prepare_role'),false);
  await f.checkEmotion('neutral',true); assert.deepEqual(f.get('emotion-picker').options.map(item=>item.value),['neutral']); assert.equal(f.get('apply-role').disabled,false);
});

test('情感数量50允许准备，51阻止提交，数组参数每项均验证', async () => {
  const emotions=Array.from({length:51},(_,index)=>({id:'e'+index,label:'情感'+index})); const state=roleState(); state.config.role={mode:'role',role_id:'many',emotion:'e0',emotions:['e0']};
  const f=fixture({state,query:name=>name==='roles'?{roles:[{id:'many',emotions}],errors:[],data_dir:'D:/data',scanned_paths:['D:/data'],upload_dir:'D:/data'}:undefined}); await f.flush();
  const checks=f.get('emotion-options').children.map(item=>item.children[0]); for(const item of checks.slice(0,50))item.checked=true; checks[49].emit('change'); await f.flush(); assert.equal(f.get('apply-role').disabled,false);
  checks[50].checked=true; checks[50].emit('change'); await f.flush(); assert.equal(f.get('apply-role').disabled,true); assert.match(f.get('role-status').textContent,/50/); assert.equal(f.calls.some(call=>call.name==='prepare_role'),false);
  checks[50].checked=false; checks[50].emit('change'); await f.flush(); await f.click('apply-role'); assert.equal(f.calls.findLast(call=>call.name==='prepare_role').args.emotions.length,50);
});

test('roleReady严格匹配默认情感、启用集合和每个profile，不接受旧单情感ready', async () => {
  for(const variant of ['missing','wrong-role','wrong-default','wrong-set']) {
    const state=roleState(); state.config.role.emotions=['happy','sad']; applyRoleState(state);
    if(variant==='missing')delete state.role_status.profiles.sad;
    if(variant==='wrong-role')state.role_status.profiles.sad.role_id='别的角色';
    if(variant==='wrong-default')state.role_status.profile.emotion='sad';
    if(variant==='wrong-set')state.role_status.emotions=['happy'];
    const f=fixture({state}); await f.flush(); assert.equal(f.get('role-load-models').disabled,true,variant); assert.equal(f.get('role-ref-text').value,'',variant);
  }
});

test('增减勾选再恢复也不复活旧profile，状态刷新同样保持dirty', async () => {
  const state=roleState(true); state.model_status.status='success'; const f=fixture({state}); await f.flush(); assert.equal(f.get('role-synthesize').disabled,false);
  await f.checkEmotion('sad',true); await f.checkEmotion('sad',false); await f.click('refresh-state'); assert.equal(f.get('role-load-models').disabled,true); assert.equal(f.get('role-synthesize').disabled,true);
  await f.click('apply-role'); await f.tick(); assert.equal(f.get('role-load-models').disabled,false);
});

test('准备成功回读state在途修改情感集合不覆盖新选项或误ready', async () => {
  let hold=false,release; const f=fixture({state:roleState(),query:(name,args,state)=>name==='state'&&hold?new Promise(resolve=>{release=()=>resolve(json(state));}):undefined}); await f.flush();
  await f.click('apply-role'); hold=true; await f.tick(); assert.ok(release); await f.checkEmotion('sad',true); await f.input('emotion-picker','sad','change'); release(); await f.flush();
  assert.deepEqual(f.get('emotion-picker').options.map(item=>item.value),['happy','sad']); assert.equal(f.get('emotion-picker').value,'sad'); assert.equal(f.get('role-load-models').disabled,true); assert.match(f.get('role-status').textContent,/准备期间.*修改/);
});

test('未测试API可准备多情感但不可加载试听，测试成功后仅手动加载', async () => {
  const state=roleState(); state.connection.status='idle'; const f=fixture({state}); await f.flush(); await f.checkEmotion('sad',true); await f.click('apply-role'); await f.tick();
  assert.equal(f.get('role-load-models').disabled,true); assert.equal(f.get('role-synthesize').disabled,true); await f.click('test-api'); assert.equal(f.get('role-load-models').disabled,false);
  assert.equal(f.calls.some(call=>call.name==='start_job'),false); await f.click('role-load-models'); await f.tick(); assert.equal(f.get('role-synthesize').disabled,false); assert.equal(f.calls.some(call=>call.name==='save_models'),false);
});

test('第四步参数与试听同页、第五步自动语音，模式使用可键盘切换的tab并持久回读', async () => {
  const state=connectedState(); state.config.voice.ref_audio_path='D:/参考.wav'; state.progress={next_step:4,message:'',completed_steps:[1,2,3,4],saved_steps:[1,2,3,4],needs_revalidate:false};
  const f=fixture({state}); await f.flush(); await f.click('step-parameters');
  assert.equal(f.get('parameters-card').hidden,false); assert.equal(f.get('mode-card').hidden,true); assert.equal(f.get('preview-card').hidden,false); assert.equal(f.get('test-card').hidden,false); assert.equal(f.get('auto-card').hidden,true); assert.equal(f.get('step-parameters').attrs.get('aria-current'),'step');
  await f.click('next-auto'); assert.equal(f.get('parameters-card').hidden,true); assert.equal(f.get('preview-card').hidden,true); assert.equal(f.get('auto-card').hidden,false); assert.equal(f.get('step-auto').attrs.get('aria-current'),'step'); assert.equal(currentStep(f),5);
  await f.click('auto-back'); assert.equal(currentStep(f),4); assert.equal(f.calls.some(call=>call.name==='start_job'),false);
  await f.click('step-models'); assert.equal(f.get('mode-selector').attrs.get('role'),'tablist');
  const event=f.get('original-mode-btn').emit('keydown',{key:'ArrowLeft'}); await f.flush(); assert.equal(event.defaultPrevented,true);
  assert.equal(f.get('role-mode-btn').attrs.get('aria-selected'),'true'); assert.equal(f.get('role-mode-btn').attrs.get('tabindex'),'0'); assert.equal(f.get('role-card').attrs.get('role'),'tabpanel'); assert.equal(f.get('role-card').hidden,false);
  assert.deepEqual(f.calls.findLast(call=>call.name==='set_mode').args,{mode:'role'}); f.get('role-mode-btn').emit('keydown',{key:'ArrowRight'}); await f.flush(); assert.equal(f.get('original-mode-btn').attrs.get('aria-selected'),'true'); assert.equal(f.get('panel-models').hidden,false);
});
test('参数读取voice与behavior、独立保存全部十字段且不污染角色指针或ready', async () => {
  const state=roleState(true); state.model_status.status='success'; Object.assign(state.config.voice,{speed_factor:1.2,top_k:8,top_p:.8,temperature:.9,seed:42,text_split_method:'cut2',parallel_infer:false}); Object.assign(state.config.behavior,{max_text_length:700,cooldown_seconds:12,send_text:true});
  const before=json({models:state.config.models,ref:state.config.voice.ref_audio_path,role:state.config.role}); const f=fixture({state}); await f.flush(); assert.equal(f.get('speed-factor').value,'1.2'); assert.equal(f.get('max-text-length').value,'700'); assert.equal(f.get('send-text').checked,true);
  await f.input('speed-factor','1.5'); await f.click('save-parameters'); const args=f.calls.findLast(call=>call.name==='save_parameters').args; assert.deepEqual(Object.keys(args).sort(),parameterKeys.slice().sort()); assert.equal(args.speed_factor,1.5); assert.equal(args.parallel_infer,false); assert.equal(args.send_text,true);
  assert.deepEqual({models:state.config.models,ref:state.config.voice.ref_audio_path,role:state.config.role},before); assert.equal(f.get('role-load-models').disabled,false); assert.equal(f.get('synthesize').disabled,false); assert.equal(f.calls.some(call=>call.name==='save_models'||call.name==='start_job'),false);
  await f.input('temperature','1.3'); await f.click('refresh-state'); assert.equal(f.get('temperature').value,'1.3'); await f.click('original-mode-btn'); assert.equal(f.get('temperature').value,'1.3'); assert.match(f.get('parameters-dirty').textContent,/未保存/);
});

test('参数严格验证数字边界整数和cut枚举、空值与后端失败保留中文错误', async () => {
  for (const [id,value] of [['speed-factor',''],['speed-factor','0.49'],['speed-factor','2.1'],['top-k','1.5'],['top-k','101'],['top-p','0'],['top-p','1.1'],['temperature','0'],['temperature','2.1'],['seed','2147483648'],['seed','-2'],['max-text-length','2001'],['max-text-length','0'],['cooldown-seconds','-1'],['cooldown-seconds','3601'],['text-split-method','cut6']]) {
    const f=fixture(); await f.flush(); await f.input(id,value); await f.click('save-parameters'); assert.equal(f.calls.some(call=>call.name==='save_parameters'),false,id+':'+value); assert.match(f.get('global-message').textContent,/操作失败/);
  }
  const f=fixture({action:name=>name==='save_parameters'?{success:false,message:'参数保存失败，请重试。'}:undefined}); await f.flush(); await f.input('speed-factor','2'); await f.click('save-parameters'); assert.match(f.get('operation-error-message').textContent,/参数保存失败/); assert.match(f.get('parameters-dirty').textContent,/未保存/);
});

test('当前配置检查无参显示backend真实摘要和问题，空schema及查询异常不破坏页面', async () => {
  const result={success:true,ready:false,message:'已选择黍，请先读取角色资源。',issues:[{level:'warning',field:'role.role_id',message:'尚未准备角色，请到第三步读取配置。'}],summary:{mode:'role',base_model:'v5turbo',role_id:'黍',emotion:'happy',emotions:['happy','sad'],api_url:'http://127.0.0.1:9880'}};
  const f=fixture({query:name=>name==='check_config'?result:undefined}); await f.flush(); await f.click('check-config'); assert.deepEqual(f.calls.findLast(call=>call.name==='check_config').args,{}); assert.match(f.get('check-config-summary').textContent,/v5turbo/); assert.match(f.get('check-config-summary').textContent,/黍/); const item=f.get('check-config-issues').children[0].children[0]; assert.equal(item.children[0].textContent,'第三步 · 声音模式 · 提醒'); assert.match(item.children[1].textContent,/尚未准备/); assert.equal(item.dataset.tone,'error'); assert.equal(item.dataset.step,'3'); assert.equal(f.get('check-config-result').dataset.tone,'error');
  for (const response of [{},null]) { const g=fixture({query:name=>name==='check_config'?response:undefined}); await g.flush(); assert.match(g.get('check-config-message').textContent,/配置检查失败.*格式不正确/); assert.equal(g.get('step-api').disabled,false); assert.equal(g.get('step-parameters').disabled,true); assert.equal(g.get('base-url').value,'http://127.0.0.1:9880'); }
  const g=fixture({query:name=>{if(name==='check_config')throw new Error('后端检查不可用');}}); await g.flush(); assert.match(g.get('check-config-message').textContent,/后端检查不可用/); assert.equal(g.get('save-parameters').disabled,false);
});

test('智能试听先必要加载再串行合成、已加载直接合成且不发QQ消息', async () => {
  const state=defaultState(); state.connection.status='success'; state.config.voice.ref_audio_path='D:/参考.wav'; const f=fixture({state}); await f.flush(); await f.click('step-parameters'); await f.click('preview-run'); assert.equal(currentStep(f),4); assert.deepEqual(f.calls.filter(call=>call.name==='start_job').map(call=>call.args.kind),['load_models']); assert.equal(f.get('preview-run').disabled,true);
  await f.tick(); assert.deepEqual(f.calls.filter(call=>call.name==='start_job').map(call=>call.args.kind),['load_models','synthesize']); assert.equal([...f.timers.values()].filter(timer=>timer.ms===1000).length,1); assert.equal(f.calls.some(call=>/qq|send_message/.test(call.name)),false);
  const ready=roleState(true); ready.model_status.status='success'; const g=fixture({state:ready}); await g.flush(); await g.click('preview-run'); assert.deepEqual(g.calls.filter(call=>call.name==='start_job').map(call=>call.args.kind),['synthesize']); assert.equal(g.calls.some(call=>call.name==='save_models'||call.name==='prepare_role'),false);
});


function requestGate() {
  const pending=[]; let enabled=true;
  return {pending, block(request) { if (enabled) return new Promise((resolve,reject)=>pending.push({...request,resolve,reject})); },
    release(name,fail=false) { const item=pending.shift(); assert.ok(item,'缺少SDK在途请求：'+name); assert.equal(item.name,name); if(fail)item.reject(new Error('模拟动作失败，不应重试'));else item.resolve(); },
    open() { enabled=false; for(const item of pending.splice(0))item.resolve(); }};
}
function assertSerial(f) { assert.equal(f.sdkConcurrency.max,1); assert.equal(f.sdkConcurrency.violations,0); assert.ok(f.sdkConcurrency.active<=1); }
function progressAt(next_step,extra={}) { return {next_step,message:'后台建议继续配置',completed_steps:Array.from({length:next_step-1},(_,i)=>i+1),needs_revalidate:false,...extra}; }
function currentStep(f) { return ['step-api','step-base','step-models','step-parameters','step-auto'].findIndex(id=>f.get(id).attrs.get('aria-current')==='step')+1; }

test('严格FIFO：boot目录、job、连续check同队列逐个放行，真实SDK从不并发',async()=>{
  const gate=requestGate(); const state=defaultState(); state.progress=progressAt(2); state.job={id:'已有任务',kind:'load_models',status:'running',message:'加载中'};
  const f=fixture({state,sdkGate:r=>gate.block(r)}); await f.flush(); gate.release('state'); await f.flush(); assert.equal(gate.pending[0].name,'catalog');
  await f.tick(); await f.click('check-config'); f.get('check-config').emit('click'); await f.flush(); assert.deepEqual(f.requests.map(r=>r.name),['state','catalog']);
  gate.release('catalog'); await f.flush(); gate.release('job'); await f.flush(); gate.release('check_config'); await f.flush(); gate.release('roles'); await f.flush();
  for(let i=0;i<5&&gate.pending.length;i++){const name=gate.pending[0].name;assert.ok(['ensure_connection','state','check_config'].includes(name));gate.release(name);await f.flush();}
  assert.equal(f.requests.filter(r=>r.name==='ensure_connection').length,1); assert.equal(f.requests.filter(r=>r.name==='job').length,1); assertSerial(f); gate.open();
});

test('动作失败不重试，FIFO继续后续查询，action后回读state不嵌套死锁',async()=>{
  const gate=requestGate(); const f=fixture({sdkGate:r=>gate.block(r)}); await f.flush(); gate.release('state'); await f.flush();
  await f.click('test-api'); await f.click('check-config'); gate.release('catalog'); await f.flush(); gate.release('save_api',true); await f.flush(); gate.release('check_config'); await f.flush(); gate.open(); await f.flush();
  assert.equal(f.requests.filter(r=>r.name==='save_api').length,1); assert.match(f.get('global-message').textContent,/操作失败/); await f.click('save-base'); assert.equal(currentStep(f),1); assert.match(f.get('global-message').textContent,/底模设置已保存。请先完成第1步「API 配置测试」/); assert.equal(f.requests.filter(r=>r.name==='state').length,2); assertSerial(f);
});

test('pagehide不执行queued mutations、不假abort active',async()=>{
  const pending=[];let hold=false;const f=fixture({sdkGate:r=>hold?new Promise(resolve=>pending.push({...r,resolve})):undefined});await f.flush();hold=true;
  await f.click('check-config');await f.click('test-api');f.dispose();await f.flush();pending.shift().resolve();await f.flush();assert.equal(f.requests.some(r=>r.name==='save_api'),false);assertSerial(f);
  let release;const g=fixture({action:name=>name==='save_api'?new Promise(resolve=>{release=resolve;}):undefined});await g.flush();await g.click('test-api');g.dispose();release({success:true});await g.flush();assert.equal(g.requests.filter(r=>r.name==='save_api').length,1);assert.equal(g.requests.some(r=>r.name==='test_api'),false);assertSerial(g);
});
test('backend progress优先恢复步骤1到5，已配自动确认不假模型就绪',async()=>{
  for(const step of [1,2,3,4,5]){const state=defaultState();state.progress=progressAt(step,{needs_revalidate:true});if(step>=4)state.config.voice.ref_audio_path='D:/已保存参考.wav';const f=fixture({state});await f.flush();assert.equal(currentStep(f),step);assert.match(f.get('global-message').textContent,step===1?/已恢复到第1步/:/已自动确认保存的接口连接/);assert.equal(f.get('load-models').disabled,step===1);assert.equal(f.get('preview-run').disabled,step===1);assert.equal(f.get('synthesize').disabled,true);assert.equal(f.requests.filter(r=>r.name==='ensure_connection').length,step===1?0:1);assert.equal(f.requests.some(r=>r.kind==='action'),false);assertSerial(f);}
});

test('init等待catalog和roles后仅恢复一次，手动导航或编辑优先且刷新不抢步',async()=>{
  for(const manual of ['navigation','edit']){const gate=requestGate();const state=withReference(defaultState());state.progress=progressAt(5);const f=fixture({state,sdkGate:r=>gate.block(r)});await f.flush();gate.release('state');await f.flush();assert.equal(currentStep(f),1);
    if(manual==='navigation')await f.click('step-parameters');else await f.input('base-url','http://正在编辑:9880');gate.release('catalog');await f.flush();if(gate.pending[0].name==='ensure_connection'){gate.release('ensure_connection');await f.flush();}assert.equal(gate.pending[0].name,'roles');assert.equal(currentStep(f),manual==='navigation'?4:1);gate.release('roles');await f.flush();gate.open();await f.flush();assert.equal(currentStep(f),manual==='navigation'?4:1);if(manual==='edit'){assert.equal(f.get('base-url').value,'http://正在编辑:9880');assert.equal(f.get('step-parameters').disabled,true);await f.input('base-url','http://127.0.0.1:9880');}await f.click('step-parameters');state.progress=progressAt(2);await f.click('refresh-state');assert.equal(currentStep(f),4);assertSerial(f);}
});

test('check更新最新progress或next_step但不跳当前编辑，继续配置按钮才跳步',async()=>{
  let next=2;const state=connectedState();state.progress=progressAt(1);const f=fixture({state,query:name=>name==='check_config'?{success:true,ready:false,message:'继续配置',issues:[],summary:{},progress:progressAt(next),next_step:5}:undefined});await f.flush();assert.equal(currentStep(f),1);await f.click('step-parameters');await f.input('speed-factor','1.4');next=3;await f.click('check-config');assert.equal(currentStep(f),4);assert.equal(f.get('speed-factor').value,'1.4');await f.click('resume-config');assert.equal(currentStep(f),3);
  const gs=connectedState();gs.progress=progressAt(2);const g=fixture({state:gs,query:name=>name==='check_config'?{success:true,ready:false,message:'前往参数',issues:[],summary:{},next_step:4}:undefined});await g.flush();assert.equal(currentStep(g),2);await g.click('resume-config');assert.equal(currentStep(g),4);assertSerial(f);assertSerial(g);
});

test('running prepare恢复第三步、load和synth恢复第四步，单轮询且刷新不再抢导航',async()=>{
  for(const kind of ['prepare_role','load_models','synthesize']){const state=roleState(kind!=='prepare_role');state.progress=progressAt(2);state.job={id:'正在执行-'+kind,kind,status:'running',message:'执行中'};const f=fixture({state});await f.flush();assert.equal(currentStep(f),kind==='prepare_role'?3:4);assert.equal([...f.timers.values()].filter(r=>r.ms===1000).length,1);await f.click('step-base');await f.click('refresh-state');assert.equal(currentStep(f),2);assert.equal([...f.timers.values()].filter(r=>r.ms===1000).length,1);assertSerial(f);}
});

test('state缺progress时保守推断API底模角色参数，不把默认custom当已完成',async()=>{
  const states=[];states.push([defaultState(),1]);const base=defaultState();base.connection.status='success';base.config.models={base_model:'custom'};states.push([base,2]);states.push([roleState(),3]);states.push([roleState(true),4]);const preview=roleState(true);Object.assign(preview.config.voice,{speed_factor:1,top_k:5,top_p:1,temperature:1,seed:-1,text_split_method:'cut5',parallel_infer:false});Object.assign(preview.config.behavior,{max_text_length:300,cooldown_seconds:5,send_text:false});states.push([preview,5]);for(const [state,step]of states){const f=fixture({state});await f.flush();assert.equal(currentStep(f),step);assertSerial(f);}
});

test('智能试听API或角色未就绪禁用，加载失败取消或配置变化不自动合成', async () => {
  const noApi=fixture(); await noApi.flush(); assert.equal(noApi.get('preview-run').disabled,true); const noRole=fixture({state:roleState()}); await noRole.flush(); assert.equal(noRole.get('preview-run').disabled,true);
  for (const status of ['error','cancelled']) { const state=defaultState(); state.connection.status='success'; state.config.voice.ref_audio_path='D:/参考.wav'; const f=fixture({state,query:(name,args,saved)=>name==='job'?{...saved.job,status,message:'加载未完成'}:undefined}); await f.flush(); await f.click('preview-run'); await f.tick(); assert.deepEqual(f.calls.filter(call=>call.name==='start_job').map(call=>call.args.kind),['load_models']); assert.match(f.get('global-message').textContent,/已停止自动试听/); }
  const state=defaultState(); state.connection.status='success'; state.config.voice.ref_audio_path='D:/参考.wav'; const f=fixture({state}); await f.flush(); await f.click('preview-run'); await f.input('top-p','.7'); await f.tick(); assert.deepEqual(f.calls.filter(call=>call.name==='start_job').map(call=>call.args.kind),['load_models']); assert.match(f.get('global-message').textContent,/配置或测试文字已改变/);
});

test('第五步（自动语音）结构、默认关闭与七字段范围和桥接声明精确', async () => {
  const auto=html.match(/<section id="auto-card"[\s\S]*?<\/section>/)?.[0]; assert.ok(auto);
  for(const id of ['auto-title','auto-enabled','auto-frequency-mode','auto-probability','auto-every-n','auto-min-interval','auto-follow-emotion','auto-keep-text','save-auto','auto-stats']) assert.match(auto,new RegExp('id="'+id+'"'));
  assert.match(html,/id="step-auto"[^>]*aria-controls="auto-card"/); assert.match(html,/id="next-auto"/);
  const binding=declaration.pages[0].actions.save_auto_voice; assert.equal(binding.api,'setup_save_auto_voice'); assert.deepEqual(Object.keys(binding.parameters).sort(),autoKeys.slice().sort());
  assert.deepEqual(binding.parameters.probability_percent,{type:'integer',required:true,minimum:0,maximum:100}); assert.deepEqual(binding.parameters.every_n,{type:'integer',required:true,minimum:1,maximum:1000}); assert.deepEqual(binding.parameters.min_interval_seconds,{type:'number',required:true,minimum:0,maximum:3600}); assert.deepEqual(binding.parameters.frequency_mode.choices,['probability','every_n']);
  const f=fixture(); await f.flush(); assert.equal(f.get('auto-enabled').checked,false); assert.equal(f.get('auto-frequency-mode').value,'probability'); assert.equal(f.get('auto-probability').value,'50'); assert.equal(f.get('auto-every-n').value,'3'); assert.equal(f.get('auto-min-interval').value,'5'); assert.equal(f.get('auto-follow-emotion').checked,true); assert.equal(f.get('auto-keep-text').checked,false); assert.equal(f.calls.some(c=>c.kind==='action'),false);
});

test('自动语音独立保存精确七字段，不发QQ不合成且读取已保存配置', async () => {
  const state=defaultState(); state.config.auto_voice={enabled:true,frequency_mode:'every_n',probability_percent:83,every_n:7,min_interval_seconds:2.5,follow_emotion:false,keep_text:true}; const before=json({models:state.config.models,voice:state.config.voice,role:state.config.role,behavior:state.config.behavior});
  const f=fixture({state}); await f.flush(); assert.equal(f.get('auto-frequency-mode').value,'every_n'); assert.equal(f.get('auto-every-n').value,'7'); assert.equal(f.get('auto-follow-emotion').checked,false); await f.input('auto-every-n','8'); await f.click('save-auto');
  const saved=f.calls.findLast(c=>c.name==='save_auto_voice'); assert.deepEqual(saved.args,{...state.config.auto_voice,every_n:8}); assert.deepEqual(Object.keys(saved.args).sort(),autoKeys.slice().sort()); assert.deepEqual({models:state.config.models,voice:state.config.voice,role:state.config.role,behavior:state.config.behavior},before); assert.deepEqual(f.calls.filter(c=>c.kind==='action').map(c=>c.name),['save_auto_voice']); assert.doesNotMatch(f.get('auto-dirty').textContent,/未保存/); assertSerial(f);
});

test('自动语音严格验证空值边界整数及枚举，0和100及极值可保存', async () => {
  for(const [id,value] of [['auto-probability',''],['auto-probability','-1'],['auto-probability','101'],['auto-probability','1.5'],['auto-every-n','0'],['auto-every-n','1001'],['auto-every-n','2.5'],['auto-min-interval',''],['auto-min-interval','-0.1'],['auto-min-interval','3600.1'],['auto-frequency-mode','invalid']]) {
    const f=fixture(); await f.flush(); await f.input(id,value); await f.click('save-auto'); assert.equal(f.calls.some(c=>c.name==='save_auto_voice'),false,id+':'+value); assert.match(f.get('global-message').textContent,/操作失败/);
  }
  for(const [probability,every,interval] of [[0,1,0],[100,1000,3600]]) { const f=fixture(); await f.flush(); await f.input('auto-probability',String(probability)); await f.input('auto-every-n',String(every)); await f.input('auto-min-interval',String(interval)); await f.click('save-auto'); assert.equal(f.calls.findLast(c=>c.name==='save_auto_voice').args.probability_percent,probability); }
  const f=fixture({action:name=>name==='save_auto_voice'?{success:false,message:'自动语音保存失败'}:undefined}); await f.flush(); await f.input('auto-probability','25'); await f.click('save-auto'); assert.match(f.get('auto-dirty').textContent,/未保存/); assert.match(f.get('operation-error-message').textContent,/第5步 · 自动语音回复：自动语音保存失败/);
});

test('自动语音编辑与签名及参数dirty独立，不清模型ready或现有试听', async () => {
  const state=roleState(true); state.model_status.status='success'; const f=fixture({state}); await f.flush(); assert.equal(f.get('synthesize').disabled,false); const pauses=f.get('audio-player').pauseCalls;
  await f.input('auto-probability','75'); assert.equal(f.get('synthesize').disabled,false); assert.equal(f.get('audio-player').pauseCalls,pauses); assert.doesNotMatch(f.get('models-dirty').textContent,/未保存/); assert.doesNotMatch(f.get('parameters-dirty').textContent,/未保存/); assert.match(f.get('auto-dirty').textContent,/未保存/);
  await f.input('speed-factor','1.5'); await f.click('save-auto'); assert.match(f.get('parameters-dirty').textContent,/未保存/); assert.doesNotMatch(f.get('auto-dirty').textContent,/未保存/); assert.equal(f.get('synthesize').disabled,false); assert.equal(f.calls.some(c=>c.name==='save_parameters'||c.name==='save_models'||c.name==='start_job'),false);
});

test('自动语音刷新和模式回读不覆盖编辑，在途保存不吃新值且FIFO串行', async () => {
  let hold=false,release; const f=fixture({query:(name,args,state)=>name==='state'&&hold?new Promise(resolve=>{release=()=>resolve(json(state));}):undefined}); await f.flush(); hold=true; await f.click('refresh-state'); await f.input('auto-probability','27'); release(); await f.flush(); hold=false; assert.equal(f.get('auto-probability').value,'27'); await f.click('role-mode-btn'); assert.equal(f.get('auto-probability').value,'27'); assert.match(f.get('auto-dirty').textContent,/未保存/);
  const pending=[]; let gated=false; const g=fixture({sdkGate:r=>gated?new Promise(resolve=>pending.push({...r,resolve})):undefined}); await g.flush(); await g.input('auto-probability','21'); gated=true; await g.click('save-auto'); await g.input('auto-probability','22'); await g.click('check-config'); assert.deepEqual(pending.map(r=>r.name),['save_auto_voice']); pending.shift().resolve(); await g.flush(); assert.deepEqual(pending.map(r=>r.name),['check_config']); pending.shift().resolve(); await g.flush(); assert.equal(g.get('auto-probability').value,'22'); assert.match(g.get('auto-dirty').textContent,/未保存/); assert.equal(g.calls.filter(c=>c.name==='save_auto_voice').length,1); assertSerial(f); assertSerial(g);
});

test('跟随情感只在role有效，保守规则已勾选默认提示及频率跳过说明完整', async () => {
  const f=fixture(); await f.flush(); assert.equal(f.get('auto-follow-emotion').disabled,true); assert.match(f.get('auto-follow-emotion-help').textContent,/普通模式.*单一 LoRA.*不选择情感/); await f.click('role-mode-btn'); assert.equal(f.get('auto-follow-emotion').disabled,false); assert.equal(f.get('auto-follow-emotion').checked,true); assert.match(f.get('auto-follow-emotion-help').textContent,/已勾选情感.*不明确.*默认/);
  assert.match(html,/优先使用消息已有的情感标记.*调用 MaiBot 的 LLM 判断（每句一次额外调用）.*保守地按关键词推断/); assert.doesNotMatch(html,/不调用额外 AI 模型/); assert.doesNotMatch(html,/AI 精准情感判断/); assert.match(html,/0% 永不转换，100% 每条/); assert.match(html,/每 N 条按 MaiBot 语义分句后的实际消息计数/); assert.match(html,/间隔内会跳过而非排队/); assert.match(html,/voice \/ image/); assert.match(html,/插件提示会跳过。语音成功后自动取消指定消息引用，避免平台拒发；转换失败保留原文字和引用。/);
});

test('auto_stats六字段安全文字摘要，刷新更新统计不覆盖自动语音编辑', async () => {
  const state=defaultState(); state.auto_stats={considered:12,converted:4,skipped:7,failed:1,last_reason:'<img src=x> 间隔未满',last_emotion:'happy'}; const f=fixture({state}); await f.flush(); assert.match(f.get('auto-stats').textContent,/已考虑：12.*已转换：4.*已跳过：7.*失败：1/); assert.match(f.get('auto-stats').textContent,/<img src=x> 间隔未满/); assert.match(f.get('auto-stats').textContent,/最近情感：happy/); assert.deepEqual(f.get('auto-stats').children,[]);
  await f.input('auto-probability','38'); state.auto_stats={considered:13,converted:5,skipped:7,failed:1,last_reason:'interval',last_emotion:'sad'}; await f.click('refresh-state'); assert.match(f.get('auto-stats').textContent,/已考虑：13.*已转换：5/); assert.match(f.get('auto-stats').textContent,/最近原因：最短间隔未满，已跳过/); assert.equal(f.get('auto-probability').value,'38');
  for(const reason of ['not_started','disabled','inactive','invalid_message','access','already_converted','duplicate','no_text','command','non_text','too_long','command_or_code','tts_error','own_speech','busy','frequency','converted','failed']) { state.auto_stats.last_reason=reason; await f.click('refresh-state'); assert.match(f.get('auto-stats').textContent.split('最近原因：')[1].split('\n')[0],chinese); assert.doesNotMatch(f.get('auto-stats').textContent,/后台诊断/); }
  state.auto_stats.last_reason='new_reason'; await f.click('refresh-state'); assert.match(f.get('auto-stats').textContent,/后台诊断：new_reason/);
});

test('第五步导航恢复与焦点，progress支持5而running加载试听恢复4', async () => {
  const state=defaultState(); state.progress=progressAt(5,{completed_steps:[1,2,3,4,5]}); state.config.voice.ref_audio_path='D:/已保存参考.wav'; const f=fixture({state}); await f.flush(); assert.equal(currentStep(f),5); assert.equal(f.get('auto-card').hidden,false); assert.equal(f.get('preview-card').hidden,true); assert.equal(f.get('auto-title').focused,true); assert.match(f.get('global-message').textContent,/已自动确认保存的接口连接/); await f.click('auto-back'); assert.equal(currentStep(f),4); assert.equal(f.get('preview-card').hidden,false); await f.click('next-auto'); assert.equal(currentStep(f),5); await f.click('step-api'); await f.click('resume-config'); assert.equal(currentStep(f),5);
  for(const kind of ['load_models','synthesize']) { const running=withReference(defaultState()); running.progress=progressAt(5); running.job={id:'恢复-'+kind,kind,status:'running',message:'任务中'}; const g=fixture({state:running}); await g.flush(); assert.equal(currentStep(g),4); assertSerial(g); }
  const invalid=defaultState(); invalid.progress=progressAt(6); const g=fixture({state:invalid}); await g.flush(); assert.equal(currentStep(g),1); assertSerial(f);
});
// Saved progress restores navigation; only a real read-only ensure restores liveness.
function reentryState(step = 4) { const state=defaultState(); state.progress=progressAt(step,{needs_revalidate:true}); state.config.voice.ref_audio_path='D:/已保存参考.wav'; return state; }
test('重入已保存配置自动确认一次，无保存加载副作用且第四步恢复可试听',async()=>{
  const f=fixture({state:reentryState()}); await f.flush();
  assert.equal(currentStep(f),4); assert.equal(f.get('preview-run').disabled,false); assert.equal(f.get('synthesize').disabled,true);
  assert.equal(f.requests.filter(r=>r.name==='ensure_connection').length,1); assert.equal(f.requests.filter(r=>r.kind==='action').length,0);
  assert.deepEqual(f.calls.find(r=>r.name==='ensure_connection').args,{});
  await f.click('step-models'); await f.click('step-parameters'); await f.click('refresh-state');
  assert.equal(f.requests.filter(r=>r.name==='ensure_connection').length,1); assertSerial(f);
});
test('重入progress markers1优先，即便nextstep1与角色runtimeidle仍自动确认连接',async()=>{
  const state=roleState(); state.connection={status:'idle',message:'运行期未确认'}; state.progress=progressAt(1,{markers:{1:true}});
  const f=fixture({state}); await f.flush(); assert.equal(f.requests.filter(r=>r.name==='ensure_connection').length,1);
  assert.equal(f.get('next-step').disabled,false); assert.equal(f.get('preview-run').disabled,true); assert.equal(f.requests.some(r=>r.name==='prepare_role'||r.name==='start_job'),false); assertSerial(f);
});
test('重入冷新无progress或插件禁用不自动确认，保存不等于连接',async()=>{
  for(const state of [defaultState(), {...defaultState(),progress:progressAt(1)}, (()=>{const s=reentryState();s.config.plugin.enabled=false;return s;})()]) {
    const f=fixture({state}); await f.flush(); assert.equal(f.requests.some(r=>r.name==='ensure_connection'),false); assert.equal(f.get('preview-run').disabled,true); assert.equal(f.get('next-step').disabled,true); assertSerial(f);
  }
});
test('重入确认失败显示具体诊断和独立重试按钮，原保存测试保留',async()=>{
  let fail=true; const f=fixture({state:reentryState(),query:name=>name==='ensure_connection'&&fail?{success:false,message:'HTTP 503 服务未启动',connection:{status:'error',message:'HTTP 503 服务未启动'}}:undefined}); await f.flush();
  assert.match(f.get('global-message').textContent,/HTTP 503/); assert.equal(f.get('retry-connection').hidden,false); assert.equal(f.get('test-api').disabled,false);
  // 确认失败后第一步未完成：返回第一步后不能再前进，重试成功才解锁，导航本身不重复确认。
  await f.click('step-api'); assert.equal(f.get('step-parameters').disabled,true); assert.match(f.get('step-parameters').title,/请先完成第1步/); assert.equal(f.requests.filter(r=>r.name==='ensure_connection').length,1);
  fail=false; await f.click('retry-connection'); assert.equal(f.get('retry-connection').hidden,true); assert.equal(f.requests.filter(r=>r.name==='ensure_connection').length,2);
  await f.click('step-parameters'); assert.equal(currentStep(f),4); assert.equal(f.get('preview-run').disabled,false); assert.equal(f.requests.filter(r=>r.name==='ensure_connection').length,2); assertSerial(f);
});
test('重入确认在途编辑API草稿不误ready，参数模型编辑也不会被回读覆盖',async()=>{
  let release; const f=fixture({state:reentryState(),query:(name,args,state)=>name==='ensure_connection'?new Promise(resolve=>{release=()=>{state.connection={status:'success',message:'确认成功',checked_url:state.config.api.base_url};resolve({success:true,message:'确认成功',connection:json(state.connection)});};}):undefined}); await f.flush();
  assert.match(f.get('synthesize-requirement').textContent,/正在自动确认连接/); await f.input('base-url','http://未保存草稿:9880'); release(); await f.flush();
  assert.equal(f.get('base-url').value,'http://未保存草稿:9880'); assert.equal(f.get('next-step').disabled,true); assert.equal(f.get('preview-run').disabled,true); assert.equal(f.get('status-step-1').dataset.tone,'error'); assert.match(f.get('status-step-1-text').textContent,/未保存/); assertSerial(f);
  let done; const g=fixture({state:reentryState(),query:(name,args,state)=>name==='ensure_connection'?new Promise(resolve=>{done=()=>{state.connection={status:'success',message:'确认成功',checked_url:state.config.api.base_url};resolve({success:true,message:'确认成功',connection:json(state.connection)});};}):undefined}); await g.flush(); await g.input('speed-factor','1.6'); await g.input('gpt-model-picker','C:/本地/语义.ckpt','change'); done(); await g.flush();
  assert.equal(g.get('speed-factor').value,'1.6'); assert.equal(g.get('gpt-model-picker').value,'C:/本地/语义.ckpt'); assert.equal(g.get('preview-run').disabled,false); assert.equal(g.get('synthesize').disabled,true); assertSerial(g);
});
test('重入确认与check并行意图仍FIFO单槽，重复导航不会重复ensure',async()=>{
  const gate=requestGate(); const f=fixture({state:reentryState(),sdkGate:r=>gate.block(r)}); await f.flush(); gate.release('state');await f.flush();gate.release('catalog');await f.flush();gate.release('roles');await f.flush();
  assert.equal(gate.pending[0].name,'ensure_connection'); f.get('check-config').emit('click'); await f.click('step-models'); await f.click('step-parameters');
  assert.equal(f.requests.filter(r=>r.name==='ensure_connection').length,1); gate.open();await f.flush();assert.equal(f.requests.filter(r=>r.name==='ensure_connection').length,1); assert.equal(f.get('preview-run').disabled,false);assertSerial(f);
});

test('第一步只保留保存并测试，第二步无补充目录与 SoVITS 路径输入，第三步无上传', () => {
  const api = html.match(/<section id="panel-api"[\s\S]*?<\/section>/)[0];
  assert.doesNotMatch(api, /仅保存接口|id="save-api"/); assert.match(api, /id="test-api"[^>]*>保存并测试连接</);
  const base = html.match(/<section id="base-card"[\s\S]*?<\/section>/)[0];
  assert.doesNotMatch(base, /补充模型目录|model-extra-dir|SoVITS 完整权重路径|id="sovits-weights"|<select id="base-model"/);
  assert.match(base, /<label for="model-root">本机模型目录<\/label>/); assert.match(base, /<label for="base-model">底模版本（自动识别）<\/label>/);
  const role = html.match(/<section id="role-card"[\s\S]*?<\/section>/)[0];
  assert.doesNotMatch(role, /type="file"|id="upload-|开始上传|上传角色资源/); assert.match(role, /请自行把角色文件夹/);
});

test('当前状态按第一二三步分类，绿色正常红色问题，点击跳转对应步骤', async () => {
  assert.match(html, /<h3 id="status-title">当前状态<\/h3>/);
  for (const [id, title] of [['status-step-1','第一步 · API 配置'],['status-step-2','第二步 · 底模'],['status-step-3','第三步 · 声音模式']]) assert.match(html, new RegExp('<button type="button" id="' + id + '"[^>]*><span class="step-status-title">' + title + '</span>'));
  assert.match(html, /\.step-status\[data-tone="success"\]\{border-color:var\(--status-ok\)/); assert.match(html, /\.step-status\[data-tone="error"\]\{border-color:var\(--status-bad\)/);
  assert.match(html, /--status-ok:#1e7e34/); assert.match(html, /--status-bad:#c5221f/); assert.doesNotMatch(html, /id="summary-(?:api|models|voice|auto)"/);
  const f = fixture({state:withReference(defaultState())}); await f.flush();
  assert.equal(f.get('status-step-1').dataset.tone, 'error'); assert.match(f.get('status-step-1-text').textContent, /尚未测试连接/);
  assert.equal(f.get('status-step-2').dataset.tone, 'success'); assert.match(f.get('status-step-2-text').textContent, /自定义权重 · 声学\.pth/);
  assert.equal(f.get('status-step-3').dataset.tone, 'success'); assert.match(f.get('status-step-3-text').textContent, /普通模式 · GPT：语义\.ckpt · 不使用 LoRA · 参考音频 平静\.wav/);
  // 第一步未完成时点击后续步骤只会停在第一步并说明原因。
  assert.equal(currentStep(f), 1); await f.click('status-step-3'); assert.equal(currentStep(f), 1); assert.match(f.get('global-message').textContent, /请先完成第1步「API 配置测试」：尚未测试连接/);
  await f.click('test-api'); assert.equal(f.get('status-step-1').dataset.tone, 'success'); assert.match(f.get('status-step-1-text').textContent, /已连接：http:\/\/127\.0\.0\.1:9880/);
  await f.click('status-step-3'); assert.equal(currentStep(f), 3); assert.equal(f.get('mode-card').hidden, false);
  await f.click('status-step-2'); assert.equal(currentStep(f), 2); assert.equal(f.get('base-card').hidden, false);
  await f.click('status-step-1'); assert.equal(currentStep(f), 1); assert.equal(f.get('panel-api').hidden, false);
  await f.input('gpt-model-picker', '', 'change'); assert.equal(f.get('status-step-3').dataset.tone, 'error'); assert.match(f.get('status-step-3-text').textContent, /未保存/);
  await f.click('save-models'); assert.equal(f.get('status-step-3').dataset.tone, 'error'); assert.match(f.get('status-step-3-text').textContent, /尚未选择 GPT 权重/);
  await f.input('lora-model-picker', 'C:/本地/角色-lora.pth', 'change'); await f.input('gpt-model-picker', 'C:/本地/语义.ckpt', 'change'); await f.click('save-models');
  assert.equal(f.get('status-step-3').dataset.tone, 'error'); assert.match(f.get('status-step-3-text').textContent, /LoRA 需要第二步识别为标准底模/);
  await f.input('sovits-model-picker', '', 'change'); assert.equal(f.get('status-step-2').dataset.tone, 'error'); assert.match(f.get('status-step-2-text').textContent, /未保存/);
  await f.input('base-url', 'http://草稿:9880'); assert.equal(f.get('status-step-1').dataset.tone, 'error'); assert.match(f.get('status-step-1-text').textContent, /未保存/);
});

test('角色模式状态：未读取显示红色并可跳到第三步，读取完成后变绿色，空角色库提示自行放入', async () => {
  const f = fixture({state:roleState()}); await f.flush();
  assert.equal(f.get('status-step-1').dataset.tone, 'success');
  assert.equal(f.get('status-step-3').dataset.tone, 'error'); assert.match(f.get('status-step-3-text').textContent, /读取角色配置/);
  await f.click('status-step-3'); assert.equal(currentStep(f), 3); assert.equal(f.get('role-card').hidden, false);
  await f.click('apply-role'); assert.equal(f.get('status-step-3').dataset.tone, 'info'); await f.tick();
  assert.equal(f.get('status-step-3').dataset.tone, 'success'); assert.match(f.get('status-step-3-text').textContent, /角色模式 · 黍 · 默认情绪：happy · 共 1 个情感/);
  const empty = fixture({state:roleState(), query:name=>name==='roles'?{roles:[],errors:[],data_dir:'C:/插件/data',scanned_paths:['C:/插件/data']}:undefined}); await empty.flush();
  assert.equal(empty.get('status-step-3').dataset.tone, 'error'); assert.match(empty.get('status-step-3-text').textContent, /放到 GPT-SoVITS 目录下的 maibot_roles/);
  assert.match(empty.get('roles-note').textContent, /请把角色文件夹放到 GPT-SoVITS 安装目录下的 maibot_roles 文件夹/);
});

test('模型目录扫描失败或已保存版本与识别结果不一致时，第二步状态为红色', async () => {
  const f = fixture({query:name=>{ if (name==='catalog') throw new Error('本机模型目录不存在或无法访问'); }}); await f.flush();
  assert.equal(f.get('status-step-2').dataset.tone, 'error'); assert.match(f.get('status-step-2-text').textContent, /^模型目录扫描失败：本机模型目录不存在或无法访问$/);
  assert.match(f.get('catalog-note').textContent, /扫描失败/);
  const state = connectedState(); Object.assign(state.config.models, {base_model:'v5turbo', sovits_weights:'C:/本地/预设.pth'});
  const g = fixture({state}); await g.flush();
  assert.equal(g.get('base-model').value, 'v2'); assert.equal(g.get('status-step-2').dataset.tone, 'error'); assert.match(g.get('status-step-2-text').textContent, /v5turbo.*v2.*不一致/);
  await g.click('status-step-2'); assert.equal(currentStep(g), 2);
});

test('操作失败原因显示在左栏最下方，可关闭且下次操作前清除', async () => {
  assert.match(html, /<section id="operation-error"[^>]*role="alert"[^>]*hidden>[\s\S]*?<\/section>\n<\/main>/);
  const f = fixture({state:connectedState(),action:name=>name==='save_parameters'?{success:false,message:'参数保存失败，请重试。'}:undefined}); await f.flush();
  assert.equal(f.get('operation-error').hidden, true);
  await f.click('step-parameters'); await f.click('save-parameters');
  assert.equal(f.get('operation-error').hidden, false); assert.equal(f.get('operation-error-message').textContent, '第4步 · 参数与试听：参数保存失败，请重试。');
  assert.match(f.get('global-message').textContent, /操作失败，原因显示在页面左栏最下方/); assert.equal(f.get('global-status').dataset.tone, 'error');
  await f.click('dismiss-operation-error'); assert.equal(f.get('operation-error').hidden, true); assert.equal(f.get('operation-error-message').textContent, '');
  await f.click('save-parameters'); assert.equal(f.get('operation-error').hidden, false);
  await f.click('save-auto'); assert.equal(f.get('operation-error').hidden, true); assert.doesNotMatch(f.get('global-message').textContent, /操作失败/);
});

test('当前配置检查：按钮操作后自动检查，绿色无问题、红色有问题，点击条目跳转所在步骤', async () => {
  let issues = [{level:'error',field:'voice.ref_audio_path',message:'普通模式尚未选择参考音频',step:3},{level:'warning',field:'api',message:'尚未在本次插件运行中测试接口连接'},{level:'info',field:'auto_voice',message:'自动语音已关闭'}];
  const f = fixture({query:(name)=>name==='check_config'?{success:true,ready:false,message:'后端提示',issues,summary:{mode:'original',base_model:'custom',role_id:'',emotion:'',emotions:[],api_url:'http://127.0.0.1:9880'}}:undefined}); await f.flush();
  const checks = () => f.calls.filter((call) => call.name === 'check_config').length;
  assert.match(html, /每次按钮操作后自动检查/); assert.equal(f.get('check-config-result').dataset.tone, 'error'); assert.match(f.get('check-config-message').textContent, /发现 2 个问题/);
  const buttons = f.get('check-config-issues').children.map((item) => item.children[0]);
  assert.deepEqual(buttons.map((button) => [button.dataset.tone, button.dataset.step]), [['error','3'],['error','1'],['success','5']]);
  assert.deepEqual(buttons.map((button) => button.children[0].textContent), ['第三步 · 声音模式 · 错误','第一步 · API 配置 · 提醒','第五步 · 自动语音回复 · 说明']);
  assert.equal(buttons[0].children[1].textContent, '普通模式尚未选择参考音频');
  // 第一步还没完成：点击第三步的问题先停在第一步并说明原因；连接成功后才能前往。
  buttons[0].emit('click'); await f.flush(); assert.equal(currentStep(f), 1); assert.match(f.get('global-message').textContent, /请先完成第1步「API 配置测试」/);
  const before = checks(); await f.click('test-api'); assert.equal(checks(), before + 1);
  const updated = f.get('check-config-issues').children.map((item) => item.children[0]);
  updated[0].emit('click'); await f.flush(); assert.equal(currentStep(f), 3);
  updated[1].emit('click'); await f.flush(); assert.equal(currentStep(f), 1);
  issues = [{level:'info',field:'role',message:'角色JSON已读取',step:3}];
  await f.click('save-parameters'); assert.equal(checks(), before + 2);
  assert.equal(f.get('check-config-result').dataset.tone, 'success'); assert.match(f.get('check-config-message').textContent, /没有发现问题/);
  assert.equal(f.get('check-config-issues').children[0].children[0].dataset.tone, 'success');
  await f.click('step-models'); await f.click('status-step-2'); assert.equal(checks(), before + 2);
  assertSerial(f);
});

test('检查进行中多次按钮操作只合并补查一次', async () => {
  const pending = []; let hold = false;
  const f = fixture({sdkGate:(request) => hold && request.name === 'check_config' ? new Promise((resolve) => pending.push({...request, resolve})) : undefined}); await f.flush();
  const count = () => f.requests.filter((request) => request.name === 'check_config').length;
  const start = count(); hold = true; await f.click('check-config'); assert.equal(count(), start + 1);
  await f.input('speed-factor', '9'); await f.click('save-parameters'); await f.click('save-parameters');
  assert.equal(f.get('operation-error').hidden, false); assert.equal(count(), start + 1);
  hold = false; pending.shift().resolve(); await f.flush();
  assert.equal(count(), start + 2); assert.equal(pending.length, 0); assertSerial(f);
});

test('当前状态共五步：参数与可选试听、自动语音按保存进度着色并可跳转', async () => {
  for (const step of [4, 5]) assert.match(html, new RegExp('<button type="button" id="status-step-' + step + '"[^>]*><span class="step-status-title">' + {4:'第四步 · 参数与试听',5:'第五步 · 自动语音回复'}[step] + '</span>'));
  const state = defaultState(); state.connection.status = 'success'; let saved = [1, 2, 3]; let preview = false;
  const f = fixture({state, query:(name)=>name==='check_config'?{success:true,ready:true,message:'ok',issues:[],summary:{},progress:{next_step:4,message:'',completed_steps:saved.filter((step) => step < 4),saved_steps:saved,preview_done:preview,needs_revalidate:false}}:undefined}); await f.flush();
  // 参考音频属于第三步：未选择时第三步为红色，第四步只提示参数尚未保存。
  assert.equal(f.get('status-step-3').dataset.tone, 'error'); assert.match(f.get('status-step-3-text').textContent, /尚未选择参考音频.*maibot_refs/);
  assert.equal(f.get('status-step-4').dataset.tone, 'error'); assert.match(f.get('status-step-4-text').textContent, /参数尚未保存过/);
  assert.equal(f.get('status-step-5').dataset.tone, 'error'); assert.match(f.get('status-step-5-text').textContent, /尚未保存自动语音设置/);
  await f.click('status-step-4'); assert.equal(currentStep(f), 3);
  await f.input('ref-audio-path', 'C:/GPT-SoVITS/maibot_refs/平静.wav', 'change'); assert.match(f.get('status-step-3-text').textContent, /GPT、LoRA 或参考音频有未保存的修改/);
  await f.click('save-models');
  const models = f.calls.findLast((call) => call.name === 'save_models').args; assert.equal(models.ref_audio_path, 'C:/GPT-SoVITS/maibot_refs/平静.wav'); assert.equal(models.gpt_weights, 'C:/模型/语义.ckpt');
  assert.equal(f.get('status-step-3').dataset.tone, 'success'); assert.match(f.get('status-step-3-text').textContent, /参考音频 平静\.wav$/);
  await f.click('next-parameters'); assert.equal(currentStep(f), 4);
  saved = [1, 2, 3, 4]; await f.click('save-parameters');
  assert.deepEqual(f.calls.filter((call) => call.kind === 'action').map((call) => call.name), ['save_models', 'save_parameters']);
  // 没有试听也是绿色：试听是可选的。
  assert.equal(f.get('status-step-4').dataset.tone, 'success'); assert.match(f.get('status-step-4-text').textContent, /^参数已保存 · 语速 1 · 按标点切分 · 试听可选，尚未试听$/);
  await f.input('speed-factor', '1.2'); assert.equal(f.get('status-step-4').dataset.tone, 'error'); assert.match(f.get('status-step-4-text').textContent, /参数有未保存的修改/);
  await f.input('speed-factor', '1');
  await f.click('status-step-5'); assert.equal(currentStep(f), 5);
  saved = [1, 2, 3, 4, 5]; await f.click('save-auto'); assert.equal(f.get('status-step-5').dataset.tone, 'success'); assert.match(f.get('status-step-5-text').textContent, /已关闭/);
  f.get('auto-enabled').checked = true; f.get('auto-enabled').emit('change'); await f.flush(); assert.equal(f.get('status-step-5').dataset.tone, 'error'); assert.match(f.get('status-step-5-text').textContent, /未保存/);
  await f.click('save-auto'); assert.equal(f.get('status-step-5').dataset.tone, 'success'); assert.match(f.get('status-step-5-text').textContent, /已启用 · 按概率 50% · 最短间隔 5 秒/);
  await f.click('status-step-4'); assert.equal(currentStep(f), 4); preview = true;
  await f.click('preview-run'); assert.equal(f.get('status-step-4').dataset.tone, 'info'); assert.match(f.get('status-step-4-text').textContent, /正在加载模型/);
  await f.tick(); await f.tick(); assert.equal(currentStep(f), 4); assert.equal(f.get('status-step-4').dataset.tone, 'success'); assert.match(f.get('status-step-4-text').textContent, /· 已试听$/);
  assertSerial(f);
});
test('第四、五步状态：角色模式不要求手填参考音频，加载失败与自动语音失败标红', async () => {
  const role = roleState(true); role.progress = {next_step:5, message:'', completed_steps:[1,2,3,4], saved_steps:[1,2,3,4], needs_revalidate:false};
  const f = fixture({state:role}); await f.flush();
  assert.equal(f.get('status-step-4').dataset.tone, 'success'); assert.match(f.get('status-step-4-text').textContent, /参数已保存 · 语速 1 · 按标点切分 · 试听可选，尚未试听/);
  const failed = defaultState(); failed.job = {id:'失败任务', kind:'load_models', status:'error', message:'HTTP 500 加载失败', audio_bytes:0, duration_seconds:0};
  failed.auto_stats = {last_reason:'failed'}; failed.config.auto_voice = {enabled:true, frequency_mode:'probability', probability_percent:50, every_n:3, min_interval_seconds:5, follow_emotion:true, keep_text:false};
  failed.config.voice.ref_audio_path = 'D:/参考.wav'; failed.progress = {next_step:5, message:'', completed_steps:[1,2,3,4], saved_steps:[1,2,3,4,5], needs_revalidate:false};
  const g = fixture({state:failed}); await g.flush();
  assert.equal(g.get('status-step-4').dataset.tone, 'error'); assert.match(g.get('status-step-4-text').textContent, /模型加载失败：HTTP 500 加载失败/);
  assert.equal(g.get('status-step-5').dataset.tone, 'error'); assert.match(g.get('status-step-5-text').textContent, /最近一次自动语音转换失败/);
});

function multiRoleState(extraRole = {}) {
  const state = roleState(); state.config.role = {...state.config.role, emotions:['happy','sad'], llm_emotion:true, llm_task:'fast_model', ...extraRole}; applyRoleState(state); return state;
}

test('LLM 情感判断桥接声明精确，元素均有中文标签', () => {
  const {queries, actions} = declaration.pages[0];
  assert.deepEqual(queries.llm_tasks, {api:'setup_llm_tasks', parameters:{}});
  assert.deepEqual(actions.save_emotion_judge, {api:'setup_save_emotion_judge', parameters:{llm_emotion:{type:'boolean',required:true}, llm_task:{type:'string',required:true,max_length:80}}});
  assert.deepEqual(actions.test_emotion, {api:'setup_test_emotion', parameters:{text:{type:'string',required:true,max_length:300}}});
  const role = html.match(/<section id="role-card"[\s\S]*?<\/section>/)[0];
  for (const id of ['llm-emotion','llm-task','refresh-llm-tasks','save-emotion-judge','emotion-test-text','test-emotion','emotion-test-result']) assert.match(role, new RegExp('id="' + id + '"'));
  assert.match(role, /用 MaiBot 的 LLM 判断每句话的情感/); assert.match(role, /LLM 失败或超时（15 秒）时使用默认情感/);
});

test('角色模式进入第三步才读取 MaiBot 模型任务一次，普通模式不读取', async () => {
  const plain = fixture({state:connectedState()}); await plain.flush(); await plain.click('step-models'); assert.equal(plain.calls.some((call) => call.name === 'llm_tasks'), false);
  const f = fixture({state:multiRoleState({llm_task:'utils'})}); await f.flush();
  assert.equal(f.calls.some((call) => call.name === 'llm_tasks'), false);
  await f.click('step-models'); await f.click('step-api'); await f.click('step-models');
  assert.equal(f.calls.filter((call) => call.name === 'llm_tasks').length, 1);
  assert.deepEqual(f.get('llm-task').options.map((option) => option.value), ['utils', 'fast_model', 'replyer']);
  assert.equal(f.get('llm-task').options[1].textContent, 'fast_model（快速模型，推荐）'); assert.equal(f.get('llm-task').value, 'utils');
  assert.equal(f.get('llm-emotion').checked, true); assert.match(f.get('emotion-judge-note').textContent, /已开启：合成前由 MaiBot 的 utils 从 2 个已启用情感中选择一个/);
  plain.get('role-mode-btn').emit('click'); await plain.flush(); assert.equal(plain.calls.filter((call) => call.name === 'llm_tasks').length, 1);
  assertSerial(f);
});

test('保存情感判断设置：未保存时第三步状态为红色，保存后为绿色并写入两字段', async () => {
  const f = fixture({state:multiRoleState()}); await f.flush(); await f.click('step-models');
  assert.equal(f.get('status-step-3').dataset.tone, 'success'); assert.match(f.get('status-step-3-text').textContent, /共 2 个情感 · LLM 判断情感（fast_model）/);
  f.get('llm-emotion').checked = false; f.get('llm-emotion').emit('change'); await f.flush();
  assert.equal(f.get('status-step-3').dataset.tone, 'error'); assert.match(f.get('status-step-3-text').textContent, /情感判断设置有未保存的修改/);
  assert.match(f.get('emotion-judge-note').textContent, /已关闭.*有未保存的修改/);
  await f.input('llm-task', 'replyer', 'change'); await f.click('save-emotion-judge');
  assert.deepEqual(f.calls.findLast((call) => call.name === 'save_emotion_judge').args, {llm_emotion:false, llm_task:'replyer'});
  assert.equal(f.state.config.role.llm_emotion, false); assert.match(f.get('global-message').textContent, /已关闭 LLM 情感判断/);
  assert.equal(f.get('status-step-3').dataset.tone, 'success'); assert.match(f.get('status-step-3-text').textContent, /未开启 LLM 判断/);
  assert.equal(f.calls.some((call) => call.name === 'start_job' || call.name === 'prepare_role'), false);
  assertSerial(f);
});

test('试判断：只有读取角色且启用 2 个以上情感才可用，有改动先保存再调用 LLM，不合成语音', async () => {
  const single = fixture({state:roleState(true)}); await single.flush(); await single.click('step-models');
  assert.equal(single.get('test-emotion').disabled, true); assert.match(single.get('emotion-judge-note').textContent, /只启用了 1 个情感.*不会调用 LLM/);
  const pending = fixture({state:(() => { const state = roleState(); state.config.role.emotions = ['happy','sad']; return state; })()}); await pending.flush();
  assert.equal(pending.get('test-emotion').disabled, true);
  assert.equal(single.get('emotion-test-result').textContent, '读取角色配置并启用 2 个以上情感后可试判断。');
  const f = fixture({state:multiRoleState()}); await f.flush(); await f.click('step-models');
  assert.equal(f.get('test-emotion').disabled, false);
  assert.equal(f.get('emotion-test-result').textContent, '输入一句话后点击“试判断”：只调用 MaiBot 的 LLM 判断情感，不合成语音。');
  await f.input('llm-task', 'utils', 'change'); await f.input('emotion-test-text', '  下雨了，有点难过  '); await f.click('test-emotion');
  assert.deepEqual(f.calls.filter((call) => call.kind === 'action').map((call) => call.name), ['save_emotion_judge', 'test_emotion']);
  assert.deepEqual(f.calls.findLast((call) => call.name === 'test_emotion').args, {text:'下雨了，有点难过'});
  assert.equal(f.get('emotion-test-result').textContent, '判断结果：开心（happy） · 模型：gpt-fast'); assert.equal(f.get('emotion-test-result').dataset.tone, 'success');
  assert.equal(f.calls.some((call) => call.name === 'start_job'), false);
  await f.input('emotion-test-text', '   '); await f.click('test-emotion'); assert.match(f.get('operation-error-message').textContent, /第3步 · 声音模式：请输入要试判断的文字/);
  assertSerial(f);
});

test('试判断失败与最近一次 LLM 判断失败都标红，并显示具体原因', async () => {
  const state = multiRoleState(); state.emotion_judge = {emotion:'happy', source:'llm_failed', model:'', detail:'RuntimeError: MaiBot LLM 调用失败：quota'};
  const f = fixture({state, action:(name) => name === 'test_emotion' ? {success:false, message:'情感判断失败：模型任务不存在'} : undefined}); await f.flush(); await f.click('step-models');
  assert.equal(f.get('status-step-3').dataset.tone, 'error'); assert.match(f.get('status-step-3-text').textContent, /最近一次 LLM 情感判断失败.*quota/);
  assert.match(f.get('emotion-judge-note').textContent, /最近一次判断失败，已使用默认情感/);
  await f.click('test-emotion');
  assert.match(f.get('emotion-test-result').textContent, /^试判断失败：情感判断失败：模型任务不存在/); assert.equal(f.get('emotion-test-result').dataset.tone, 'error');
  assert.equal(f.get('operation-error').hidden, false); assert.match(f.get('operation-error-message').textContent, /第3步 · 声音模式：情感判断失败：模型任务不存在/);
  const ok = multiRoleState(); ok.emotion_judge = {emotion:'sad', source:'llm', model:'gpt-fast', detail:''};
  const g = fixture({state:ok}); await g.flush(); assert.match(g.get('emotion-judge-note').textContent, /最近一次判断：难过（gpt-fast）/); assert.equal(g.get('status-step-3').dataset.tone, 'success');
});

test('刷新模型任务失败显示在左栏底部，不影响已保存的任务选择', async () => {
  let fail = true; const f = fixture({state:multiRoleState({llm_task:'utils'}), query:(name) => { if (name === 'llm_tasks' && fail) return {success:false, tasks:[], message:'无法读取 MaiBot 模型任务：未授权'}; }}); await f.flush();
  await f.click('step-models'); assert.match(f.get('emotion-judge-note').textContent, /模型任务列表读取失败：无法读取 MaiBot 模型任务：未授权/);
  assert.equal(f.get('llm-task').value, 'utils'); assert.equal(f.get('operation-error').hidden, true);
  await f.click('refresh-llm-tasks'); assert.equal(f.get('operation-error').hidden, false); assert.match(f.get('operation-error-message').textContent, /第3步 · 声音模式：无法读取 MaiBot 模型任务：未授权/);
  fail = false; await f.click('refresh-llm-tasks'); assert.match(f.get('global-message').textContent, /已读取 MaiBot 的 3 个模型任务/); assert.doesNotMatch(f.get('emotion-judge-note').textContent, /读取失败/);
  assert.equal(f.get('llm-task').value, 'utils');
});

test('自动语音跟随情感提示说明多情感时由 LLM 判断', async () => {
  const state = multiRoleState(); state.progress = {next_step:5,message:'',completed_steps:[1,2,3,4],saved_steps:[1,2,3,4],needs_revalidate:false};
  const f = fixture({state}); await f.flush(); assert.equal(currentStep(f), 5);
  assert.match(f.get('auto-follow-emotion-help').textContent, /由 MaiBot 的 LLM 判断每句话的情感/);
  await f.click('step-models'); f.get('llm-emotion').checked = false; f.get('llm-emotion').emit('change'); await f.flush();
  assert.doesNotMatch(f.get('auto-follow-emotion-help').textContent, /LLM/);
});

test('第一步下方有 GPT-SoVITS 部署教程：整合包、源码安装、启动 API 与常见问题', () => {
  const api = html.match(/<section id="panel-api"[\s\S]*?<\/section>/)[0];
  const guide = api.match(/<details id="deploy-guide" class="guide">[\s\S]*?<\/details>/)?.[0];
  assert.ok(guide, '第一步缺少部署教程');
  assert.match(guide, /<summary>还没有部署 GPT-SoVITS？查看部署教程<\/summary>/);
  for (const text of ['GPT-SoVITS-v5-20261005.7z', 'GPT-SoVITS-v5-20261005nvidia50x0.7z', 'pwsh -F install.ps1 --Device CU128 --Source HF',
    'bash install.sh --device CU128 --source HF', 'python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml',
    'runtime\\python.exe api_v2.py', 'http://127.0.0.1:9880/docs', 'device: cpu', 'is_half: false', '-a 0.0.0.0', '9872、9874'])
    assert.ok(guide.includes(text), text);
  for (const url of ['https://huggingface.co/lj1995/GPT-SoVITS-windows-package/resolve/main/GPT-SoVITS-v5-20261005.7z',
    'https://huggingface.co/lj1995/GPT-SoVITS-windows-package/resolve/main/GPT-SoVITS-v5-20261005nvidia50x0.7z',
    'https://huggingface.co/lj1995/GPT-SoVITS-windows-package/tree/main', 'https://github.com/RVC-Boss/GPT-SoVITS'])
    assert.ok(guide.includes('<code class="url">' + url + '</code>'), url);
  assert.match(guide, /无法直接打开外部链接：点击链接文字即可全选/);
});

test('第二步下方有底模选择建议、各版本所需文件与下载地址', () => {
  const base = html.match(/<section id="base-card"[\s\S]*?<\/section>/)[0];
  const guide = base.match(/<details id="base-guide" class="guide">[\s\S]*?<\/details>/)?.[0];
  assert.ok(guide, '第二步缺少底模说明');
  assert.match(guide, /<summary>该选哪个底模？底模下载地址<\/summary>/);
  const rows = [...guide.matchAll(/<tr><td>([^<]+)<\/td>/g)].map((match) => match[1]);
  assert.deepEqual(rows, ['v5turbo', 'v5dev', 'v2ProPlus', 'v2Pro', 'v4', 'v3', 'v2', 'v1']);
  for (const file of ['gsv-v5-pretrained/s2Gv5turbo.pth', 'gsv-v5-pretrained/vocoder.pth', 'v2Pro/s2Gv2Pro.pth', 'sv/pretrained_eres2netv2w24s4ep4.ckpt',
    'gsv-v4-pretrained/s2Gv4.pth', 'models--nvidia--bigvgan_v2_24khz_100band_256x/', 'gsv-v2final-pretrained/s2G2333k.pth', 's2G488k.pth', 'chinese-hubert-base/', 'G2PWModel'])
    assert.ok(guide.includes(file), file);
  for (const url of ['https://huggingface.co/lj1995/GPT-SoVITS/tree/main', 'https://huggingface.co/lj1995/GPT-SoVITS/tree/main/gsv-v5-pretrained',
    'https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/gsv-v5-pretrained/s2Gv5turbo.pth',
    'https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/gsv-v5-pretrained/vocoder.pth',
    'https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/s1v3.ckpt',
    'https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/pretrained_models.zip',
    'https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip',
    'https://github.com/RVC-Boss/GPT-SoVITS/wiki'])
    assert.ok(guide.includes('<code class="url">' + url + '</code>'), url);
  assert.match(guide, /v1～v4、v2Pro 打包下载（约 4\.3 GB，不含 v5/); assert.match(guide, /使用 Windows 整合包时底模已经齐全/);
});

test('两份教程只保留 GitHub 和 Hugging Face 链接（本机接口地址除外）', () => {
  for (const id of ['deploy-guide', 'base-guide']) {
    const guide = html.match(new RegExp('<details id="' + id + '" class="guide">[\\s\\S]*?<\\/details>'))[0];
    const urls = [...guide.matchAll(/https?:\/\/[^\s<"'）)，。、]+/g)].map((match) => match[0]);
    assert.ok(urls.length > 0, id);
    for (const url of urls)
      assert.match(url, /^(https:\/\/(github\.com|huggingface\.co)\/|http:\/\/127\.0\.0\.1:9880)/, id + ': ' + url);
    assert.doesNotMatch(guide, /modelscope|ModelScope|魔搭|hf-mirror|yuque|语雀|docker\.com|百度网盘|iCloud/, id);
    for (const code of guide.matchAll(/<code class="url">([^<]+)<\/code>/g))
      assert.match(code[1], /^(https:\/\/(github\.com|huggingface\.co)\/|http:\/\/127\.0\.0\.1:9880)/, id + ': ' + code[1]);
  }
});

test('教程链接只作为可复制文字，不生成会离开设置页的链接', () => {
  const anchors = [...html.matchAll(/<a\b[^>]*>/g)].map((match) => match[0]);
  assert.deepEqual(anchors.map((tag) => tag.match(/id="([^"]+)"/)?.[1] || tag.match(/href="([^"]+)"/)?.[1]), ['#main', 'download-audio']);
  assert.match(html, /\.guide code\.url\{user-select:all/); assert.match(html, /\.guide pre\{[^}]*user-select:all/);
});

test('连接测试失败自动展开部署教程，扫描不到底模或扫描失败自动展开底模说明', async () => {
  const f = fixture({action:(name) => name === 'test_api' ? {success:false, message:'连接失败'} : undefined}); await f.flush();
  assert.notEqual(f.get('deploy-guide').open, true); await f.click('test-api'); assert.equal(f.get('deploy-guide').open, true);
  const ok = fixture(); await ok.flush(); assert.notEqual(ok.get('base-guide').open, true);
  const empty = fixture({query:(name) => name === 'catalog' ? {base_models:[], gpt_weights:[], sovits_weights:[], lora_weights:[], note:'未找到模型'} : undefined}); await empty.flush();
  assert.equal(empty.get('base-guide').open, true);
  const broken = fixture({query:(name) => { if (name === 'catalog') throw new Error('目录不存在'); }}); await broken.flush();
  assert.equal(broken.get('base-guide').open, true);
});

test('每一步配置完成才能进入下一步：未完成时导航锁定、下一步禁用，已到达的步骤随时可返回', async () => {
  let saved = [];
  const f = fixture({query:(name, args, state)=>name==='check_config'?{success:true,ready:false,message:'检查',issues:[],summary:{},progress:{next_step:Math.min(saved.length + 1, 5),message:'',completed_steps:saved,saved_steps:saved,needs_revalidate:false}}:undefined}); await f.flush();
  assert.equal(currentStep(f), 1);
  for (const id of ['step-base', 'step-models', 'step-parameters', 'step-auto', 'next-step']) assert.equal(f.get(id).disabled, true, id);
  assert.equal(f.get('step-auto').title, '请先完成第1步「API 配置测试」');
  await f.click('status-step-4'); assert.equal(currentStep(f), 1); assert.match(f.get('global-message').textContent, /请先完成第1步「API 配置测试」：尚未测试连接.*每一步配置完成后才能进入下一步/);
  await f.click('test-api');
  // 底模已保存（第二步完成）；普通模式还没选参考音频，第三步未完成：最远只能到第三步。
  for (const id of ['step-base', 'step-models', 'next-step']) assert.equal(f.get(id).disabled, false, id);
  for (const id of ['step-parameters', 'step-auto']) assert.equal(f.get(id).disabled, true, id);
  assert.equal(f.get('step-auto').title, '请先完成第3步「声音模式」');
  await f.click('next-step'); assert.equal(currentStep(f), 2);
  await f.input('model-root', 'D:/未保存目录');
  assert.equal(f.get('step-models').disabled, true); assert.match(f.get('step-models').title, /请先完成第2步「选择底模」/);
  assert.equal(f.get('step-api').disabled, false); assert.equal(f.get('base-back-step').disabled, false);
  await f.input('model-root', 'C:/GPT-SoVITS'); assert.equal(f.get('step-models').disabled, false);
  await f.click('step-models'); assert.equal(currentStep(f), 3); assert.equal(f.get('next-parameters').disabled, true);
  await f.input('ref-audio-path', 'C:/GPT-SoVITS/maibot_refs/平静.wav', 'change'); assert.equal(f.get('next-parameters').disabled, true);
  saved = [1, 2, 3]; await f.click('save-models'); assert.equal(f.get('next-parameters').disabled, false);
  await f.input('gpt-model-picker', '', 'change');
  assert.equal(f.get('next-parameters').disabled, true); assert.equal(f.get('step-parameters').disabled, true);
  assert.equal(f.get('back-step').disabled, false);
  await f.input('gpt-model-picker', 'C:/模型/语义.ckpt', 'change'); await f.click('next-parameters'); assert.equal(currentStep(f), 4);
  assert.equal(f.get('next-auto').disabled, true);
  saved = [1, 2, 3, 4]; await f.click('save-parameters'); assert.equal(f.get('next-auto').disabled, false);
  await f.click('next-auto'); assert.equal(currentStep(f), 5);
  // 回到前面的步骤不受限制。
  await f.click('step-api'); assert.equal(currentStep(f), 1); await f.click('step-auto'); assert.equal(currentStep(f), 5);
});

test('恢复进度不会越过未完成的步骤，并说明需要先完成的内容', async () => {
  const state = roleState(); state.progress = {next_step:5,message:'后台建议继续配置',completed_steps:[1,2,3,4],saved_steps:[1,2,3,4],needs_revalidate:false};
  const f = fixture({state}); await f.flush();
  assert.equal(currentStep(f), 3);
  assert.match(f.get('global-message').textContent, /已恢复到第3步：声音模式。这一步还没有完成：尚未读取当前角色配置.*完成后才能进入下一步/);
  for (const id of ['step-parameters', 'step-auto', 'next-parameters']) assert.equal(f.get(id).disabled, true, id);
  await f.click('apply-role'); await f.tick();
  assert.equal(f.get('step-parameters').disabled, false); assert.equal(f.get('step-auto').disabled, false);
  await f.click('next-parameters'); assert.equal(currentStep(f), 4);
});

test('第二步未选择底模时“保存底模并继续”不可点击：未扫描到权重时提示先扫描，选中后才可保存', async () => {
  const state = connectedState(); state.config.models = {...state.config.models, gpt_weights:'', sovits_weights:''};
  const f = fixture({state, query:(name)=>name==='catalog'?{base_models:[],gpt_weights:[],sovits_weights:[],lora_weights:[],note:'未找到模型'}:undefined}); await f.flush();
  await f.click('step-base'); assert.equal(currentStep(f), 2);
  assert.equal(f.get('save-base').disabled, true); assert.match(f.get('base-dirty').textContent, /请先填写本机模型目录并扫描，再选择 SoVITS 底模或完整权重/);
  const g = fixture({state:(() => { const s = connectedState(); s.config.models = {...s.config.models, gpt_weights:'', sovits_weights:''}; return s; })()}); await g.flush();
  await g.click('step-base'); assert.equal(g.get('save-base').disabled, true);
  assert.match(g.get('base-dirty').textContent, /请先在上方下拉框中选择/);
  await g.input('sovits-model-picker', 'C:/本地/预设.pth', 'change'); assert.equal(g.get('save-base').disabled, false);
  await g.click('save-base'); assert.deepEqual(g.calls.findLast((call) => call.name === 'save_base').args, {root:'C:/GPT-SoVITS', sovits_weights:'C:/本地/预设.pth'}); assert.equal(currentStep(g), 3);
});

test('第三步普通模式用下拉框选择 maibot_refs 中的参考音频，保存后第三步才完成', async () => {
  const f = fixture({state:connectedState(defaultState())}); await f.flush();
  const options = f.get('ref-audio-path').options.map((option) => [option.value, option.textContent]);
  assert.deepEqual(options.slice(0, 3), [['', '请选择参考音频'], ['C:/GPT-SoVITS/maibot_refs/平静.wav', '平静.wav · maibot_refs · C:/GPT-SoVITS/maibot_refs/平静.wav'], ['C:/GPT-SoVITS/maibot_refs/开心/开心.wav', '开心.wav · maibot_refs/开心 · C:/GPT-SoVITS/maibot_refs/开心/开心.wav']]);
  assert.equal(f.get('ref-audio-path').value, 'C:/GPT-SoVITS/maibot_refs/平静.wav');
  assert.equal(f.get('ref-audio-full').textContent, '完整路径：C:/GPT-SoVITS/maibot_refs/平静.wav');
  assert.match(f.get('refs-note').textContent, /找到 2 个参考音频/);
  assert.equal(f.get('refs-dir-hint').textContent, 'C:/GPT-SoVITS/maibot_refs'); assert.equal(f.get('roles-dir-hint').textContent, 'C:/GPT-SoVITS/maibot_roles');
  // 保存在 maibot_refs 之外的路径仍保留为“已保存”选项。
  const g = fixture({state:withReference(connectedState(), 'D:/其他目录/外部.wav')}); await g.flush();
  assert.ok(g.get('ref-audio-path').options.some((option) => option.value === 'D:/其他目录/外部.wav' && /^已保存：外部\.wav/.test(option.textContent)));
  await f.click('step-models'); assert.equal(currentStep(f), 3); assert.equal(f.get('status-step-3').dataset.tone, 'success');
  await f.input('ref-audio-path', 'C:/GPT-SoVITS/maibot_refs/开心/开心.wav', 'change');
  assert.equal(f.get('status-step-3').dataset.tone, 'error'); assert.equal(f.get('next-parameters').disabled, true);
  await f.click('save-models'); assert.equal(f.calls.findLast((call) => call.name === 'save_models').args.ref_audio_path, 'C:/GPT-SoVITS/maibot_refs/开心/开心.wav');
  assert.match(f.get('status-step-3-text').textContent, /参考音频 开心\.wav$/); assert.equal(f.get('next-parameters').disabled, false);
  const before = f.calls.filter((call) => call.name === 'catalog').length; await f.click('refresh-refs');
  assert.equal(f.calls.filter((call) => call.name === 'catalog').length, before + 1); assert.match(f.get('global-message').textContent, /已重新扫描 GPT-SoVITS 目录下的 maibot_refs/);
});

test('第三步下方说明参考音频放到 GPT-SoVITS 的 maibot_refs、角色放到 maibot_roles，没有可选文件时自动展开', async () => {
  const mode = html.match(/<section id="mode-card"[\s\S]*?<details id="voice-guide" class="guide">[\s\S]*?<\/details>/)?.[0];
  assert.ok(mode, '第三步缺少放置说明');
  const guide = mode.slice(mode.indexOf('<details id="voice-guide"'));
  for (const text of ['<summary>参考音频和角色文件放在哪里？</summary>', 'maibot_refs', 'maibot_roles/角色名/role.json', '3～10 秒', 'v3、v4、v5 底模必填', '不要放到插件的 data 目录', '本页不提供上传'])
    assert.ok(guide.includes(text), text);
  assert.match(guide, /<code id="refs-dir-hint" class="url">/); assert.match(guide, /<code id="roles-dir-hint" class="url">/);
  assert.doesNotMatch(html, /插件 data 目录后刷新|复制到本插件固定 data 目录|插件角色目录（固定）/);
  const empty = fixture({state:connectedState(), query:(name) => name === 'catalog' ? {base_models:[], gpt_weights:[{label:'GPT', path:'C:/模型/语义.ckpt'}], sovits_weights:[{label:'SoVITS', path:'C:/模型/声学.pth'}], lora_weights:[], note:'扫描完成', reference_audios:[], reference_dir:'C:/GPT-SoVITS/maibot_refs', reference_note:'参考音频文件夹不存在：C:/GPT-SoVITS/maibot_refs。请在 GPT-SoVITS 安装目录下新建 maibot_refs 文件夹并放入参考音频。'} : undefined}); await empty.flush();
  assert.equal(empty.get('voice-guide').open, true); assert.match(empty.get('refs-note').textContent, /参考音频文件夹不存在/);
  const ok = fixture({state:connectedState()}); await ok.flush(); assert.notEqual(ok.get('voice-guide').open, true);
});

test('第五步运行摘要：角色模式尚无情感时只写“暂无”，普通模式说明不选择情感', async () => {
  const role = fixture({state:roleState(true)}); await role.flush();
  assert.match(role.get('auto-stats').textContent, /最近情感：暂无$/); assert.doesNotMatch(role.get('auto-stats').textContent, /普通模式/);
  const original = fixture(); await original.flush();
  assert.match(original.get('auto-stats').textContent, /最近情感：暂无（普通模式不选择情感）$/);
  const recent = roleState(true); recent.auto_stats = {last_emotion:'sad'}; const f = fixture({state:recent}); await f.flush();
  assert.match(f.get('auto-stats').textContent, /最近情感：sad$/);
});
