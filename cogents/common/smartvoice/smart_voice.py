"""
Cogent Smart Voice Implementation
Merges Aliyun Lingjie AI transcription and text extraction logic
"""

import json
import logging
import time
from typing import Any, Dict, Optional

from aliyunsdkcore.acs_exception.exceptions import ClientException, ServerException
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.request import CommonRequest

from .base_smart_voice import SmartVoiceBase


class SmartVoice(SmartVoiceBase):
    """
    Cogent Smart Voice implementation using Aliyun Lingjie AI service.
    Handles both transcription API calls and text extraction.
    """

    def __init__(self, ak_id: str, ak_secret: str, app_key: str, region_id: str = "cn-shanghai") -> None:
        """
        Initialize SmartVoice with Aliyun credentials.

        Args:
            ak_id: Aliyun Access Key ID
            ak_secret: Aliyun Access Key Secret
            app_key: NLS App Key
            region_id: Aliyun region ID (default: cn-shanghai)
        """
        self.ak_id = ak_id
        self.ak_secret = ak_secret
        self.app_key = app_key
        self.region_id = region_id

        # Constants for Aliyun NLS service
        self.PRODUCT = "nls-filetrans"
        self.DOMAIN = f"filetrans.{region_id}.aliyuncs.com"
        self.API_VERSION = "2018-08-17"
        self.POST_REQUEST_ACTION = "SubmitTask"
        self.GET_REQUEST_ACTION = "GetTaskResult"

        # Request parameters
        self.KEY_APP_KEY = "appkey"
        self.KEY_FILE_LINK = "file_link"
        self.KEY_VERSION = "version"
        self.KEY_ENABLE_WORDS = "enable_words"
        self.KEY_AUTO_SPLIT = "auto_split"

        # Response parameters
        self.KEY_TASK = "Task"
        self.KEY_TASK_ID = "TaskId"
        self.KEY_STATUS_TEXT = "StatusText"
        self.KEY_RESULT = "Result"

        # Status values
        self.STATUS_SUCCESS = "SUCCESS"
        self.STATUS_RUNNING = "RUNNING"
        self.STATUS_QUEUEING = "QUEUEING"

        # Create AcsClient instance
        self.client = AcsClient(ak_id, ak_secret, region_id)

    def transcribe(self, audio_data: str) -> Optional[str]:
        """
        Transcribe audio data to text.

        Args:
            audio_data: URL of the audio file to transcribe

        Returns:
            Transcribed text or None if transcription fails
        """
        # First, perform the transcription
        result = self._transcribe_file(audio_data)
        if result is None:
            return None

        # Then extract the text from the result
        return self._extract_transcription_text(result)

    def _transcribe_file(self, file_link: str) -> Optional[Dict[str, Any]]:
        """
        Perform file transcription using Aliyun NLS service.

        Args:
            file_link: URL of the audio file to transcribe

        Returns:
            Transcription result dictionary or None if failed
        """
        # Submit transcription request
        post_request = CommonRequest()
        post_request.set_domain(self.DOMAIN)
        post_request.set_version(self.API_VERSION)
        post_request.set_product(self.PRODUCT)
        post_request.set_action_name(self.POST_REQUEST_ACTION)
        post_request.set_method("POST")

        # Configure task parameters
        # Use version 4.0 for new integrations, set enable_words to False by default
        task = {
            self.KEY_APP_KEY: self.app_key,
            self.KEY_FILE_LINK: file_link,
            self.KEY_VERSION: "4.0",
            self.KEY_ENABLE_WORDS: False,
        }

        # Uncomment to enable auto split for multi-speaker scenarios
        # task[self.KEY_AUTO_SPLIT] = True

        task_json = json.dumps(task)
        logging.info(f"Submitting task: {task_json}")
        post_request.add_body_params(self.KEY_TASK, task_json)

        task_id = ""
        try:
            post_response = self.client.do_action_with_exception(post_request)
            post_response_json = json.loads(post_response)
            logging.info(f"Submit response: {post_response_json}")

            status_text = post_response_json[self.KEY_STATUS_TEXT]
            if status_text == self.STATUS_SUCCESS:
                logging.info("File transcription request submitted successfully!")
                task_id = post_response_json[self.KEY_TASK_ID]
            else:
                logging.error(f"File transcription request failed: {status_text}")
                return None
        except ServerException as e:
            logging.error(f"Server error: {e}")
            return None
        except ClientException as e:
            logging.error(f"Client error: {e}")
            return None

        if not task_id:
            logging.error("No task ID received")
            return None

        # Create request to get task result
        get_request = CommonRequest()
        get_request.set_domain(self.DOMAIN)
        get_request.set_version(self.API_VERSION)
        get_request.set_product(self.PRODUCT)
        get_request.set_action_name(self.GET_REQUEST_ACTION)
        get_request.set_method("GET")
        get_request.add_query_param(self.KEY_TASK_ID, task_id)

        # Poll for results
        logging.info(f"Polling for results with task ID: {task_id}")
        status_text = ""
        max_attempts = 60  # Maximum 10 minutes (60 * 10 seconds)
        attempt = 0

        while attempt < max_attempts:
            try:
                get_response = self.client.do_action_with_exception(get_request)
                get_response_json = json.loads(get_response)
                logging.info(f"Poll response (attempt {attempt + 1}): {get_response_json}")

                status_text = get_response_json[self.KEY_STATUS_TEXT]
                if status_text == self.STATUS_RUNNING or status_text == self.STATUS_QUEUEING:
                    # Continue polling
                    time.sleep(10)
                    attempt += 1
                else:
                    # Exit polling
                    break
            except ServerException as e:
                logging.error(f"Server error during polling: {e}")
                return None
            except ClientException as e:
                logging.error(f"Client error during polling: {e}")
                return None

        if status_text == self.STATUS_SUCCESS:
            logging.info("File transcription completed successfully!")
            return get_response_json.get(self.KEY_RESULT)
        else:
            logging.error(f"File transcription failed with status: {status_text}")
            return None

    def _extract_transcription_text(self, result: Dict[str, Any]) -> Optional[str]:
        """
        Extract transcription text from the lingji_ai result.

        Args:
            result: The result from transcribe_file function

        Returns:
            Extracted transcription text or None if extraction fails
        """
        try:
            # The result structure from lingji_ai contains sentences with text
            if isinstance(result, dict) and "Sentences" in result:
                sentences = result["Sentences"]
                if isinstance(sentences, list):
                    # Extract text from each sentence, avoiding duplicates
                    # Since there are duplicate entries with different ChannelId,
                    # we'll use a set to store unique texts
                    unique_texts = set()
                    for sentence in sentences:
                        if isinstance(sentence, dict) and "Text" in sentence:
                            text = sentence["Text"].strip()
                            if text:  # Only add non-empty text
                                unique_texts.add(text)

                    # Convert set back to list and join
                    if unique_texts:
                        transcription_parts = sorted(list(unique_texts))
                        return " ".join(transcription_parts)

            # If the structure is different, try to find text in the result
            if isinstance(result, dict):
                # Look for common transcription result keys
                for key in ["text", "transcription", "content", "result"]:
                    if key in result:
                        return str(result[key])

                # If no direct text found, try to extract from nested structure
                result_str = json.dumps(result, ensure_ascii=False)
                # This is a fallback - return the full result as string
                return result_str

            # If result is already a string, return it
            if isinstance(result, str):
                return result

        except Exception as e:
            logging.error(f"Error extracting transcription text: {str(e)}")
            return None

        return None
