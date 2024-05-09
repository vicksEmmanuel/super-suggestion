import asyncio
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

router = APIRouter()


@router.post("/stream")
async def stream_data(request: Request):
    """An example of a streaming response. This endpoint will return
    a stream of numbers from 0 to num, with a 1 second pause
    between each number."""
    body = await request.json()
    num = body.get('num', 10)  # get the number of items to generate from the request body, default is 10

    async def content_generator():
        """Generator function to generate the stream of numbers"""
        for i in range(num):
            yield str(i) + "\n"
            await asyncio.sleep(0.1)  # pause for a 0.1 seconds

    return StreamingResponse(content_generator(), media_type='text/plain')
