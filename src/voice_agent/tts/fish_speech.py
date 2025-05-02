from __future__ import annotations

import asyncio

from livekit.agents import (
    APITimeoutError,
    APIConnectOptions,
    APIConnectionError,
    tts,
    utils,
)
from speechlab.tts.fish_speech.client import FishSpeechRequest, FishSpeechAsyncClient

_SAMPLE_RATE: int = 44100
_NUM_CHANNELS: int = 1


class FishSpeech(tts.TTS):
    def __init__(
        self,
        endpoint: str,
        reference_id: str | None = None,
        seed: int | None = None,
        max_new_tokens: int = 1024,
        top_p: float = 0.9,
        temperature: float = 0.6,
        repetition_penalty: float = 1.2,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=False,
            ),
            sample_rate=_SAMPLE_RATE,
            num_channels=_NUM_CHANNELS,
        )
        self._session_id = 0
        self._client = FishSpeechAsyncClient(
            endpoint,
            client_id=f"voice-agent-{id(self)}",
            name="FishSpeech",
        )
        self._def_request = FishSpeechRequest(
            session_id=-1,
            text="",
            reference_id=reference_id,
            seed=seed,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )

    def _make_request(self, text: str) -> FishSpeechRequest:
        self._session_id += 1
        return self._def_request.model_copy(
            update={
                "session_id": self._session_id,
                "text": text,
            }
        )

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions | None = None,
    ) -> ChunkedStream:
        return ChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
            request=self._make_request(text),
        )

    def _close_client(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    async def aclose(self) -> None:
        self._close_client()
        await super().aclose()


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: FishSpeech,
        input_text: str,
        request: FishSpeechRequest,
        conn_options: APIConnectOptions | None = None,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._client = tts._client
        self._request = request

    async def _run(self) -> None:
        request_id = utils.shortuuid()
        audio_bstream = utils.audio.AudioByteStream(
            sample_rate=self._tts.sample_rate,
            num_channels=self._tts.num_channels,
        )

        try:
            iterator = self._client.tts(self._request)
            async for bytes_data in iterator:
                for frame in audio_bstream.write(bytes_data):
                    self._event_ch.send_nowait(
                        tts.SynthesizedAudio(
                            request_id=request_id,
                            frame=frame,
                        )
                    )

            for frame in audio_bstream.flush():
                self._event_ch.send_nowait(
                    tts.SynthesizedAudio(
                        request_id=request_id,
                        frame=frame,
                    )
                )
        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except Exception as e:
            raise APIConnectionError() from e
