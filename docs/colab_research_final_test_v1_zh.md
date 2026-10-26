# 四模型最终test v1：Colab命令书（当前不要开启GPU）

先读 [冻结附录](research_final_test_plan_v1_zh.md)。**目前仅准备；请先把Colab实际余额告知，再确定硬预算。** 不运行旧训练命令、不续训、不选择新checkpoint。此包与旧GRPO训练包不同。

## 0. 本地（无需Colab/GPU）

```bash
python -m scripts.eval_research_final_test_v1
```

默认只读原冻结字节/锁和240行计数，`test_labels_parsed=false`，不加载官方权重、不跑test。包附带本轮CPU验收报告。以下Colab格均未在本地实际执行；收到余额并确认预算后才用。

## 1. 批准后：Drive与独立包

新建同版本A100环境，不与旧ART/vLLM环境混用。上传本次 `final_test_v1_bundle.zip`，核对提供的ZIP SHA256后执行：

```python
from google.colab import drive, files
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, signal, subprocess, sys, zipfile

drive.mount('/content/drive')
repo = Path('/content/AgenticRL-final-test-v1')
repo.mkdir(exist_ok=False)
upload = files.upload()
assert len(upload) == 1
name = next(iter(upload))
expected_bundle_sha = input('粘贴随包提供的SHA256：').strip()
assert len(expected_bundle_sha) == 64
assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected_bundle_sha
with zipfile.ZipFile(name) as z:
    assert len(z.namelist()) == len(set(z.namelist()))
    assert all((repo/n).resolve().is_relative_to(repo.resolve()) for n in z.namelist())
    z.extractall(repo)
bundle = json.loads((repo/'bundle_manifest.json').read_text())
assert bundle['version'] == 'four-model-final-test-v1'
assert all(hashlib.sha256((repo/n).read_bytes()).hexdigest() == h for n,h in bundle['files'].items())
run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
local = Path('/content/final-test-results')/run_id
backup = Path('/content/drive/MyDrive/AgenticRL-final-test-v1')/run_id
# 此时不创建local/backup：入口必须独占新目录；旧证据绝不删除。
```

## 2. 独立环境、固定Base下载、CPU合同

```python
subprocess.run([sys.executable,'-m','pip','install','-q','uv'],check=True)
venv = Path('/content/final-test-venv')
subprocess.run(['uv','venv','--python','3.12',str(venv)],check=True)
python = venv/'bin/python'
subprocess.run(['uv','pip','install','--python',str(python),'-r',str(repo/'requirements-sft-pilot.txt')],check=True)
r = json.loads((repo/'final_test_base_restore.json').read_text())
code = """
import json,sys
from huggingface_hub import snapshot_download
r=json.load(open(sys.argv[1]))
p=snapshot_download('Qwen/Qwen2.5-0.5B-Instruct',revision=r['revision'],
    cache_dir='/root/.cache/huggingface/hub',allow_patterns=list(r['base_files']))
print(p)
"""
p = subprocess.run([str(python),'-c',code,str(repo/'final_test_base_restore.json')],check=True,text=True,capture_output=True)
base = Path(p.stdout.strip().splitlines()[-1])
cpu_env = dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
subprocess.run([str(python),'-m','pytest','tests/test_research_final_test_v1.py','tests/test_research_final_test_campaign_v1.py','-q'],cwd=repo,env=cpu_env,check=True)
subprocess.run([str(python),'-m','scripts.eval_research_final_test_v1'],cwd=repo,env=cpu_env,check=True)
# 上面仅下载/字节检查和validation派生的scripted CPU fixture；不读取test标签做调试。
```

## 3. 核对余额并显式执行（预算尚待确认）

不得直接填旧“剩余12h−5.29h”。1.80h仅是历史step10 validation速度的机械外推，不保证test用时；建议讨论至少30%评测余量及安装/同步空间，实际硬限必须由余额决定。余额不足不运行。不通过缩短4096上下文来凑预算。

```python
balance_note = input('记录刚刚在Colab界面确认的实际剩余额度/时间、确认时间：').strip()
assert balance_note
cap = int(input('输入双方已确认的本次评测进程硬限秒数（不含此前安装/idle）：'))
assert cap > 420
approval = input('确认四模型×240，固定协议、不训练、失败停止，输入 RUN_FROZEN_FOUR_MODEL_TEST_960：')
assert approval == 'RUN_FROZEN_FOUR_MODEL_TEST_960'
gpu = subprocess.check_output(['nvidia-smi'],text=True)
assert 'A100' in gpu
cmd = [str(python),'-m','scripts.eval_research_final_test_v1',
       '--run-gpu','--confirm',approval,'--balance-note',balance_note,'--wall-seconds',str(cap),
       '--base-dir',str(base),'--drive-root','/content/drive/MyDrive',
       '--output',str(local),'--backup',str(backup)]
env = dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
log_path = Path('/content')/f'{run_id}-final-test.log'
rc = None
with log_path.open('x') as log:
    proc = subprocess.Popen(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        rc = proc.wait(timeout=cap)  # 外部硬限包括预检/加载/评测/同步，不只检查任务间边界
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGTERM)
        try: proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL)
            proc.wait()
        print('硬预算中止，未完成，先回传，不自动续评。')
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL)
                proc.wait()
        print('exit=',proc.returncode)
print(log_path.read_text()[-5000:])
print(subprocess.check_output(['nvidia-smi'],text=True))
```

评测中可另开**只读**监控格：`print(log_path.read_text()[-3000:])`。不启动第二个进程。

入口提前420秒停止启动新任务以争取保存空间，但不保证底层CUDA超时可被Python取消；外层watchdog可能再用10秒清理进程，不能将其称为零开销硬实时保证。OOM/infra失败后终止整个cohort；保留失败和原协议，不缩短/跳题。若GPU仍有本次进程，关闭/删除运行时。

`exit0 + manifest completed`才是完整结束。`exit2 + paused-clean-boundary`允许在**重新确认余额且不查看成绩调参**后用同一cmd加 `--resume` 继续，外部watchdog规则相同、cap/余额记录需更新。入口先重放全部完成题并核Drive，已经完成题不再调用模型。`failed-stop-investigate/running/loading`和硬kill一律先回传，不自行重试。不新建run冒充第一次test。

## 4. 成功或失败都反馈；成功必须读盘核对960条

```python
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

check = {'local_exists':local.exists(),'backup_exists':backup.exists(),'matched':False,'counts':{},'complete':False}
if local.exists() and backup.exists():
    left = {p.relative_to(local).as_posix():sha(p) for p in local.rglob('*') if p.is_file()}
    right = {p.relative_to(backup).as_posix():sha(p) for p in backup.rglob('*') if p.is_file()}
    check['matched'] = left == right
    m = json.loads((local/'manifest.json').read_text())
    for arm in ['SFT30','GRPO20261011','GRPO20261012','GRPO20261013']:
        paths = sorted((local/arm/'attempts').glob('*.json'))
        rows = [json.loads(p.read_text()) for p in paths]  # 实际重新打开，而非相信打印
        check['counts'][arm] = {'files':len(paths),'unique_attempts':len({r['attempt_id'] for r in rows}),
                                'infra':sum(r['infra_failure'] for r in rows)}
    check['complete'] = (check['matched'] and m['state']=='completed' and
        all(c == {'files':240,'unique_attempts':240,'infra':0} for c in check['counts'].values()))
    if m['state']=='completed': assert check['complete'],'保存/数量不一致，不能报告完整test'
print(json.dumps(check,ensure_ascii=False,indent=2))
feedback = Path('/content')/f'{run_id}-final-test-feedback.zip'
with zipfile.ZipFile(feedback,'x',compression=zipfile.ZIP_DEFLATED) as z:
    for p in local.rglob('*'):
        if p.is_file(): z.write(p,'run/'+p.relative_to(local).as_posix())
    z.writestr('readback.json',json.dumps(check,ensure_ascii=False,indent=2))
    if log_path.exists(): z.write(log_path,'execution.log')
    z.writestr('nvidia_after.txt',subprocess.check_output(['nvidia-smi'],text=True))
    z.write(repo/'configs/research/final_test_v1.json','final_test_v1.json')
    z.write(repo/'configs/research/final_test_v1.lock.json','final_test_v1.lock.json')
    z.write(repo/'bundle_manifest.json','bundle_manifest.json')
print(feedback,feedback.stat().st_size,sha(feedback))
# 反馈包无模型/优化器。成功完整或失败readback=false都原样回传，不删失败证据。
import shutil
backup.parent.mkdir(parents=True,exist_ok=True)
shutil.copyfile(feedback,backup.parent/feedback.name)
assert sha(feedback)==sha(backup.parent/feedback.name)
files.download(str(feedback))
```

把反馈放到本地 `results/` 后再做最终复算与结论。结束确认Drive原模型与逐题结果保留，及时释放GPU。不得只回传summary或只挑最好seed；日志缺口和LP局限也继续保留。
