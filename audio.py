import requests
import pyaudio


class SVCAudioWebInterface:
    def __init__(self, host: str ="localhost", port: int=9880) -> None:
        self.url = f"http://{host}:{port}/tts"
        self.session = pyaudio.PyAudio()
        self.stream = self.session.open(format=pyaudio.paInt16,
                                   channels=2,
                                   rate=16000,
                                   output=True)
        self.text_lang = "zh"
    
    def process(self, text: str) -> bool:
        # 构造http请求
        request_data = {
            "text": text,
            "text_lang": self.text_lang,
            "ref_audio_path": "noelle_sample.wav",
            "prompt_text": "嗯！希望这种甜蜜的果子，能为你们酿出最甜最好的酒。",
            "prompt_lang": "zh",
            "streaming_mode": True,
            "parallel_infer": True,
        }
        # 从 HTTP 获取音频流
        response = requests.post(self.url, json=request_data, stream=True)

        # 检查请求是否成功
        if response.status_code == 200:
            # print("开始播放音频...")
            try:
                # 持续读取音频数据块并播放
                for chunk_data in response.iter_content(chunk_size=1024):
                    if chunk_data:
                        self.stream.write(chunk_data)
            except KeyboardInterrupt:
                print("停止播放")
        else:
            print(f"请求失败，状态码: {response.status_code}")
        return True
    
    def __del__(self):
        self.stream.stop_stream()
        self.stream.close()
        self.session.terminate()


def main():
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
    
    for s in text:
        audio.process(s)        


if __name__ == "__main__":
    main()
