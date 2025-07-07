# agent_launcher.py
import subprocess

running_agents = {}

def launch_agent(keyword: str):
    """단일 키워드로 에이전트 실행"""
    if keyword in running_agents:
        print(f"[SKIP] '{keyword}' 에이전트는 이미 실행 중.")
        return

    proc = subprocess.Popen(["python", "agent_worker.py", "--keyword", keyword])
    running_agents[keyword] = proc
    print(f"[LAUNCHER] '{keyword}' 에이전트 실행됨.")

def launch_agents(keywords: list[str]):
    """다수 키워드로 에이전트 실행"""
    for keyword in keywords:
        launch_agent(keyword)