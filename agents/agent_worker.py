# agent_worker.py
import argparse
import time

def run_agent(keyword: str):
    print(f"[{keyword}] 전문가 에이전트 시작됨.")
    while True:
        time.sleep(5)
        print(f"[{keyword}] 대기 중...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", required=True)
    args = parser.parse_args()

    run_agent(args.keyword)
