import pygame
import random
import time
import csv
from tkinter import filedialog, Tk
from pynput import keyboard

# 초기화
pygame.init()

# 색상 설정
RED = (255, 0, 0)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
DARK_GRAY = (150, 150, 150)
DARK = (0, 0, 0)

# 화면 크기 설정
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# 한글 폰트 설정 - 폰트 파일 경로를 정확히 지정하세요!
font_path = "C:/Users/Rick/Desktop/EEG(최신)/stoop task/NanumGothic-Bold.ttf"  # 폰트 파일 경로
font_size = 150  # 폰트 크기
font = pygame.font.Font(font_path, font_size)

# 태스크 조건 설정
TASKS = [
    {"text": "출발", "color": GREEN, "marker_id": 23},
    {"text": "정지", "color": RED, "marker_id": 24},
    {"text": "출발", "color": RED, "marker_id": 25},
    {"text": "정지", "color": GREEN, "marker_id": 26}
]

# 태스크 시작 여부
start_task = False
results = []
base_time = None

# correct와 incorrect의 응답속도를 추적하는 변수 추가
correct_count = 0
incorrect_count = 0  # 여기에 추가
correct_response_times = []
incorrect_response_times = []

def save_to_csv():
    root = Tk()
    root.withdraw()
    save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if not save_path:
        return

    with open(save_path, 'w', newline='') as csvfile:
        fieldnames = ["latency", "duration", "type", "marker_value", "key", "timestamp", "marker_id", "correct_count", "incorrect_count", "correct_average_response_time", "incorrect_average_response_time"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"Data saved to {save_path}")

# 전역 키 리스너 설정
slash_pressed = False
comma_pressed = False

def on_key_press(key):
    global start_task, base_time, slash_pressed, comma_pressed
    if key == keyboard.KeyCode.from_char('1'):
        if base_time is None:  # 처음 시작되면 base_time 설정
            base_time = time.perf_counter()
        start_task = True
    elif key == keyboard.KeyCode.from_char('/'):
        slash_pressed = True
    elif key == keyboard.KeyCode.from_char(','):
        comma_pressed = True

def on_key_release(key):
    global slash_pressed, comma_pressed
    if key == keyboard.KeyCode.from_char('/'):
        slash_pressed = False
    elif key == keyboard.KeyCode.from_char(','):
        comma_pressed = False
listener = keyboard.Listener(on_press=on_key_press, on_release=on_key_release)
listener.start()

# 프레임 속도 제한을 위한 Clock 객체 생성
clock = pygame.time.Clock()

running = True
while running:
    screen.fill(DARK)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            continue

    if start_task:
        if len(results) >= 10:
            running = False
            continue

        # 대기 시간 설정
        wait_time = random.uniform(1.5, 2.0)
        time.sleep(wait_time)

        # Calculate latency here
        latency = time.perf_counter() - base_time  # 초 단위로 변환

        correct_answer = random.random() <= 0.5
        if correct_answer:
            # 50% 확률 중에서도 50% 확률로 '출발' 또는 '정지'를 선택
            task = TASKS[2] if random.random() <= 0.5 else TASKS[3]
        else:
             # 나머지 50% 확률로 '출발', '정지' 중 하나 선택
            task = TASKS[0] if random.random() <= 0.5 else TASKS[1]

        text_surface = font.render(task["text"], True, task["color"])
        screen.blit(text_surface, (SCREEN_WIDTH // 2 - text_surface.get_width() // 2, SCREEN_HEIGHT // 2 - text_surface.get_height() // 2))
        pygame.display.flip()

        time.sleep(0.1)

        screen.fill(DARK)
        pygame.display.flip()

        duration = None
        start_time = time.perf_counter()
        while (time.perf_counter() - start_time) < 1.5:
            if slash_pressed or comma_pressed:
                duration = time.perf_counter() - start_time
                break
            time.sleep(0.01)  # Sleep for a short duration to prevent 100% CPU utilization

        # Correct 조건에 따른 추가적인 처리
        if task["marker_id"] == 25 and slash_pressed:
            type_result = "correct"
            correct_count += 1
            if duration is not None:
                correct_response_times.append(duration)
           
        elif task["marker_id"] == 26 and comma_pressed:
            type_result = "correct"
            correct_count += 1
            if duration is not None:
                correct_response_times.append(duration)

        elif task["marker_id"] in [23, 24] and not (slash_pressed or comma_pressed):
            type_result = "correct"
            correct_count += 1
            if duration is not None:
                correct_response_times.append(duration)
        else:
            type_result = "incorrect"
            incorrect_count += 1
            if duration is not None:
                incorrect_response_times.append(duration)

        results.append({
            "latency": latency,
            "duration": duration,
            "type": type_result,
            "marker_value": None,
            "key": None,
            "timestamp": None,
            "marker_id": task["marker_id"],
            "incorrect_count": incorrect_count,
            "correct_count": correct_count,
            "correct_average_response_time": sum(correct_response_times) / len(correct_response_times) if correct_response_times else None,
            "incorrect_average_response_time": sum(incorrect_response_times) / len(incorrect_response_times) if incorrect_response_times else None
        })
    pygame.display.flip()
    clock.tick(60)  # 프레임 속도를 60FPS로 제한

pygame.quit()
save_to_csv()
