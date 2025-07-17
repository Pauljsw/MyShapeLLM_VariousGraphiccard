import subprocess
import os
import json

# 고정된 shapellm 가상환경의 python 경로
SHAPELLM_PYTHON = "/home/aimgroup/anaconda3/envs/shapellm/bin/python"

def run_shape_llm(pts_file, prompt, model_path="qizekun/ShapeLLM_13B_general_v1.0", conv_mode="llava_sw"):
    # CLI 실행 기준 디렉토리 (ShapeLLM/)
    shape_llm_dir = os.path.dirname(__file__)
    cli_path = os.path.join("llava", "serve", "cli.py")
    abs_pts_file = os.path.abspath(pts_file)

    # 명령어 구성
    cmd = [
        SHAPELLM_PYTHON,
        cli_path,
        "--model-path", model_path,
        "--pts-file", abs_pts_file,
        "--conv-mode", conv_mode
    ]

    # PYTHONPATH 설정 추가
    env = os.environ.copy()
    env["PYTHONPATH"] = shape_llm_dir

    try:
        process = subprocess.Popen(
            cmd,
            cwd=shape_llm_dir,        # cli.py 실행 위치
            env=env,                  # PYTHONPATH 포함
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(input=prompt + "\n", timeout=180)
    except subprocess.TimeoutExpired:
        process.kill()
        return {"status": "error", "message": "⏱️ ShapeLLM execution timed out", "raw_output": "", "target": abs_pts_file, "prompt": prompt, "location_type": "unknown"} # location_type 추가
    except Exception as e:
        return {"status": "error", "message": f"ShapeLLM Subprocess error: {str(e)}", "raw_output": "", "target": abs_pts_file, "prompt": prompt, "location_type": "unknown"} # location_type 추가


    if process.returncode != 0:
        return {"status": "error", "message": f"ShapeLLM CLI execution error: {stderr.strip()}", "raw_output": stderr.strip(), "target": abs_pts_file, "prompt": prompt, "location_type": "unknown"} # location_type 추가


    raw_output = stdout.strip()
    print(f"[DEBUG] ShapeLLM Raw Output: {raw_output}")

    status = "uncertain"
    location_type = "unknown"
    # reasoning_steps = "No reasoning provided." # 이 줄 제거
    message = raw_output

    try:
        # 모델의 출력이 JSON 코드 블록 안에 있을 수 있으므로, JSON만 추출
        json_start = raw_output.find('```json')
        json_end = raw_output.find('```', json_start + 1)
        if json_start != -1 and json_end != -1:
            json_str = raw_output[json_start + len('```json'):json_end].strip()
        else:
            json_str = raw_output.strip()


        parsed_result = json.loads(json_str)

        # 'status' 필드 확인
        if "status" in parsed_result and parsed_result["status"] in ["required", "not_required", "uncertain"]:
            status = parsed_result["status"]
        else:
            print(f"[WARNING] JSON output 'status' field is missing or invalid. Falling back to 'uncertain'. Output: {raw_output}")

        # 'location_type' 필드 파싱
        if "location_type" in parsed_result and parsed_result["location_type"] in ["exterior", "interior", "unknown"]:
            location_type = parsed_result["location_type"]
        else:
            print(f"[WARNING] JSON output 'location_type' field is missing or invalid. Falling back to 'unknown'. Output: {raw_output}")

        # 'reasoning_steps' 필드 파싱 로직 제거
        # if "reasoning_steps" in parsed_result:
        #     reasoning_steps = parsed_result["reasoning_steps"]
        # else:
        #     print(f"[WARNING] JSON output 'reasoning_steps' field is missing. Falling back to default message. Output: {raw_output}")

        # 'reason' 필드 확인 (message로 사용)
        if "reason" in parsed_result:
            message = parsed_result["reason"]
        else:
            print(f"[WARNING] JSON output 'reason' field is missing. Falling back to raw output. Output: {raw_output}")


    except json.JSONDecodeError:
        print(f"[ERROR] ShapeLLM output is not valid JSON. Falling back to defaults. Output: {raw_output}")
        status = "uncertain"
        location_type = "unknown"
        # reasoning_steps = "Failed to parse JSON." # 이 줄 제거
        message = raw_output
    except Exception as e:
        print(f"[ERROR] Error processing ShapeLLM JSON output: {e}. Falling back to defaults. Output: {raw_output}")
        status = "uncertain"
        location_type = "unknown"
        # reasoning_steps = f"Error in processing: {str(e)}" # 이 줄 제거
        message = raw_output

    return {
        "status": status,
        "location_type": location_type,
        # "reasoning_steps": reasoning_steps, # 이 줄 제거
        "target": abs_pts_file,
        "prompt": prompt,
        "raw_output": raw_output,
        "message": message
    }