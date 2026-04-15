import subprocess

def auto_commit_and_push(commit_msg="Auto-commit by MLOps Agent"):
    try:
        # 모든 파일 스테이징
        subprocess.run(['git', 'add', '.'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 커밋 (변경점이 없으면 예외 발생 가능)
        subprocess.run(['git', 'commit', '-m', commit_msg], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print(f"[MLOps] 로컬 Git 마일스톤 저장 완료: {commit_msg}")
        
        # 추후 리모트 레포지토리가 등록되면 아래 주석을 풉니다.
        # subprocess.run(['git', 'push', 'origin', 'main'], check=True)
        # print("[MLOps] 깃허브 원격 저장소 푸시 완료!")
        
    except subprocess.CalledProcessError as e:
        print(f"[MLOps] Git 커밋/푸시 중 오류가 발생했거나 변경사항이 없습니다.")
