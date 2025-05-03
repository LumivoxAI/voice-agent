from speechlab.preprocess.runorm.client import RuNormRequest, RuNormAsyncClient


class RuNorm:
    def __init__(self, endpoint: str) -> None:
        self._session_id = 0
        self._client = RuNormAsyncClient(
            endpoint,
            client_id=f"voice-agent-{id(self)}",
            name="RuNorm",
        )

    async def __call__(self, text: str) -> str:
        self._session_id += 1
        # Heuristically, it works better that way
        text = text[:-1] if text.endswith(".") else text
        req = RuNormRequest(
            session_id=self._session_id,
            text=text,
        )
        return await self._client.preprocess(req)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
