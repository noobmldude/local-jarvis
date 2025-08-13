#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketParams
from pipecat.transports.services.daily import DailyParams

from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.services.cartesia.tts import CartesiaTTSService
import aiohttp
from pipecat.services.piper.tts import PiperTTSService

from pipecat.services.ollama.llm import OLLamaLLMService

from pipecat.utils.text.markdown_text_filter import MarkdownTextFilter
from pipecat.audio.interruptions.min_words_interruption_strategy import (
    MinWordsInterruptionStrategy,
)
from pipecat.observers.loggers.llm_log_observer import LLMLogObserver

strategy = MinWordsInterruptionStrategy(min_words=3)

load_dotenv(override=True)


# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
}


async def run_example(
    transport: BaseTransport, _: argparse.Namespace, handle_sigint: bool
):
    logger.info(f"Starting bot")

    # stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    #
    # tts = DeepgramTTSService(
    #     api_key=os.getenv("DEEPGRAM_API_KEY"), voice="aura-2-andromeda-en"
    # )
    #

    stt = WhisperSTTService()

    # tl = TranscriptionLogger()

    md_filter = MarkdownTextFilter(
        params=MarkdownTextFilter.InputParams(filter_code=True, filter_tables=True)
    )
    # tts = CartesiaTTSService(
    #     api_key=os.getenv("CARTESIA_API_KEY"),
    #     voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    #     # voice_id=os.getenv("CARTESIA_VOICE_ID", "4d2fd738-3b3d-4368-957a-bb4805275bd9"),
    #     # British Narration Lady: 4d2fd738-3b3d-4368-957a-bb4805275bd9
    #     sample_rate=24000,  # Optional: specify sample rate
    #     text_filter=md_filter,
    # )

    # Create aiohttp session
    session = aiohttp.ClientSession()

    # Configure service
    tts = PiperTTSService(
        # base_url="http://localhost:5151/api/tts",
        base_url="http://localhost:5151",
        aiohttp_session=session,
        sample_rate=22050,  # Optional: specify if you know the model's sample rate
        text_filter=md_filter,
    )

    # llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))
    llm = OLLamaLLMService(model="gemma3:1b", base_url="http://localhost:11434/v1")

    messages = [
        {
            "role": "system",
            "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters or markdown formatting in your answers. Respond to what the user said in a creative and helpful way.",
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,  # STT
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            interruption_strategies=[MinWordsInterruptionStrategy(min_words=3)],
            # enable_metrics=True,
            # enable_usage_metrics=True,
            # observers=[LLMLogObserver()],  # Log Ollama LLM interactions
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        messages.append(
            {"role": "system", "content": "Please introduce yourself to the user."}
        )
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=handle_sigint)

    await runner.run(task)


if __name__ == "__main__":
    from pipecat.examples.run import main

    main(run_example, transport_params=transport_params)
