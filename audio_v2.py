import asyncio
import aiohttp
import pyaudio
from threading import Thread


class SVCAudioWebInterface:
    def __init__(self, host: str = "localhost", port: int = 9880) -> None:
        self.url = f"http://{host}:{port}/tts"
        self.text_lang = "zh"
        self.task_queue = asyncio.Queue()  # 用于接收任务
        self.result_queue = asyncio.Queue()  # 用于保存已完成的音频数据
        self.running = True

        # 初始化音频播放
        self.session = pyaudio.PyAudio()
        self.stream = self.session.open(
            format=pyaudio.paInt16,
            channels=2,
            rate=16000,
            output=True
        )

        # 启动任务处理线程和结果处理线程
        self.task_thread = Thread(target=self._task_handler, daemon=True)
        self.result_thread = Thread(target=self._result_handler, daemon=True)
        self.task_thread.start()
        self.result_thread.start()

    def _task_handler(self):
        """任务处理线程，负责发送请求并读取音频数据"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)  # 设置当前线程的事件循环
        loop.run_until_complete(self._process_tasks())

    async def _process_tasks(self):
        """监听任务队列并发送 HTTP 请求"""
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout()) as session:
            while self.running:
                if not self.task_queue.empty():
                    task = await self.task_queue.get()  # 获取任务
                    text = task["text"]
                    request_data = {
                        "text": text,
                        "text_lang": self.text_lang,
                        "ref_audio_path": "noelle_sample.wav",
                        "prompt_text": "嗯！希望这种甜蜜的果子，能为你们酿出最甜最好的酒。",
                        "prompt_lang": "zh",
                        "streaming_mode": False,  # 关闭流式传输
                        "parallel_infer": True,
                    }

                    # 发送请求并将响应句柄存入结果队列
                    try:
                        response = await session.post(self.url, json=request_data)
                        if response.status == 200:
                            print(f"请求成功: {text}")
                            # 收集音频数据
                            audio_data = b""
                            async for chunk_data in response.content.iter_chunked(1024):
                                audio_data += chunk_data
                            # 将完整音频数据传递给结果队列
                            await self.result_queue.put({"audio_data": audio_data, "text": text})
                        else:
                            print(f"请求失败，状态码: {response.status}")
                    except Exception as e:
                        print(f"请求错误: {e}")

    def _result_handler(self):
        """结果处理线程，负责播放请求结果"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)  # 设置当前线程的事件循环
        loop.run_until_complete(self._process_results())

    async def _process_results(self):
        """监听结果队列并播放音频"""
        while self.running:
            if not self.result_queue.empty():
                result = await self.result_queue.get()  # 等待获取结果
                audio_data = result["audio_data"]
                print(f"开始播放任务: {result['text']}")
                self.stream.write(audio_data)  # 播放完整音频数据

    async def add_task(self, text: str):
        """将播放任务添加到任务队列"""
        await self.task_queue.put({"text": text})
        print(f"任务已添加：{text}")

    def stop(self):
        """停止播放和关闭资源"""
        self.running = False
        self.task_thread.join()
        self.result_thread.join()
        self.stream.stop_stream()
        self.stream.close()
        self.session.terminate()


async def main():
    audio = SVCAudioWebInterface()
    audio.text_lang = "all_ja"

    text = [
        "ノエルは信じています。",
        "どんなに辛く苦しいことがあって，",
        "秋君が负けそうになってしまっても，",
        "世界中の谁も秋君を信じなくなって，",
        "秋君自身も自分のことが信じなれなくとしても，",
        "ノエルは信じています，",
        "ノエルを救ってくれた秋君が本物の英雄なんだって。 "
    ]

    # 异步添加任务
    for s in text:
        await audio.add_task(s)

    while True:
        pass
    audio.stop()


if __name__ == "__main__":
    asyncio.run(main())
