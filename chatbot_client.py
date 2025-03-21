#!/usr/bin/env python3
"""
Enhanced chatbot client for interacting with the chatbot service.
Supports both test client and real endpoint access, as well as multi-turn conversations.
"""

import os
import json
import logging
import requests
from typing import Dict, Any, Optional, List, Union, Tuple
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ChatbotClient:
    """
    An enhanced client for interacting with the chatbot service API.
    Supports both test client and real endpoint access, as well as multi-turn conversations.
    """
    
    def __init__(
        self, 
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 30,
        test_client=None,
        conversation_id_field: str = "conversation_id",
        history_field: str = "history"
    ):
        """
        Initialize the ChatbotClient.
        
        Args:
            endpoint: The chatbot API endpoint URL
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            test_client: Optional test client for direct integration testing
            conversation_id_field: Field name for conversation ID in requests/responses
            history_field: Field name for conversation history in requests/responses
        """
        self.endpoint = endpoint or os.getenv("CHATBOT_API_ENDPOINT", "http://localhost:8000/chat")
        self.api_key = api_key or os.getenv("CHATBOT_API_KEY", "")
        self.timeout = timeout
        self.test_client = test_client
        self.conversation_id_field = conversation_id_field
        self.history_field = history_field
        
        # Store active conversations
        self.active_conversations = {}
        
        logger.info(f"Initialized ChatbotClient with endpoint: {self.endpoint}")
        if test_client:
            logger.info("Using test client for API requests")
    
    def get_headers(self) -> Dict[str, str]:
        """
        Get the headers for API requests.
        
        Returns:
            Dictionary of request headers
        """
        headers = {
            "Content-Type": "application/json"
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        return headers
    
    def query(self, message: str, conversation_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Send a query to the chatbot API.
        
        Args:
            message: The message to send to the chatbot
            conversation_id: Optional conversation ID for multi-turn conversations
            **kwargs: Additional parameters to include in the request
            
        Returns:
            Dictionary containing the API response
        """
        headers = self.get_headers()
        
        # Prepare payload
        payload = {
            "message": message,
            **kwargs
        }
        
        # Add conversation ID if provided
        if conversation_id:
            payload[self.conversation_id_field] = conversation_id
            
            # Add conversation history if available
            if conversation_id in self.active_conversations:
                payload[self.history_field] = self.active_conversations[conversation_id]
        
        try:
            logger.info(f"Sending query to chatbot API: {message[:50]}...")
            
            if self.test_client:
                # Use test client for API requests
                response = self.test_client.post(
                    self.endpoint,
                    headers=headers,
                    json=payload
                )
                response_data = response.json()
            else:
                # Use requests for API requests
                response = requests.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                response.raise_for_status()
                response_data = response.json()
            
            # Update conversation history if conversation ID is provided
            if conversation_id:
                if conversation_id not in self.active_conversations:
                    self.active_conversations[conversation_id] = []
                
                # Add the current exchange to the conversation history
                self.active_conversations[conversation_id].append({
                    "user": message,
                    "assistant": self.extract_response(response_data)
                })
            
            return response_data
        
        except Exception as e:
            logger.error(f"Error querying chatbot API: {str(e)}")
            raise
    
    def extract_response(self, response_data: Dict[str, Any]) -> str:
        """
        Extract the response text from the API response data.
        
        Args:
            response_data: The API response data
            
        Returns:
            The extracted response text
        """
        # Try different common response fields
        if "response" in response_data:
            return response_data["response"]
        elif "answer" in response_data:
            return response_data["answer"]
        elif "message" in response_data:
            return response_data["message"]
        elif "text" in response_data:
            return response_data["text"]
        elif "content" in response_data:
            return response_data["content"]
        else:
            # If no standard field is found, return the stringified response
            logger.warning("Could not find response text in standard fields, using full response")
            return json.dumps(response_data)
    
    def query_and_extract(self, message: str, conversation_id: Optional[str] = None, **kwargs) -> str:
        """
        Send a query and extract the response text.
        
        Args:
            message: The message to send to the chatbot
            conversation_id: Optional conversation ID for multi-turn conversations
            **kwargs: Additional parameters to include in the request
            
        Returns:
            The extracted response text
        """
        response_data = self.query(message, conversation_id, **kwargs)
        return self.extract_response(response_data)
    
    def start_conversation(self) -> str:
        """
        Start a new conversation and return the conversation ID.
        
        Returns:
            New conversation ID
        """
        import uuid
        conversation_id = str(uuid.uuid4())
        self.active_conversations[conversation_id] = []
        return conversation_id
    
    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, str]]:
        """
        Get the history of a conversation.
        
        Args:
            conversation_id: The conversation ID
            
        Returns:
            List of conversation exchanges
        """
        return self.active_conversations.get(conversation_id, [])
    
    def clear_conversation(self, conversation_id: str) -> bool:
        """
        Clear the history of a conversation.
        
        Args:
            conversation_id: The conversation ID
            
        Returns:
            True if the conversation was cleared, False if it didn't exist
        """
        if conversation_id in self.active_conversations:
            self.active_conversations[conversation_id] = []
            return True
        return False
    
    def end_conversation(self, conversation_id: str) -> bool:
        """
        End a conversation and remove it from active conversations.
        
        Args:
            conversation_id: The conversation ID
            
        Returns:
            True if the conversation was ended, False if it didn't exist
        """
        if conversation_id in self.active_conversations:
            del self.active_conversations[conversation_id]
            return True
        return False
    
    def multi_turn_conversation(self, messages: List[str], **kwargs) -> Tuple[str, List[Dict[str, str]]]:
        """
        Conduct a multi-turn conversation with the chatbot.
        
        Args:
            messages: List of messages to send in sequence
            **kwargs: Additional parameters to include in the requests
            
        Returns:
            Tuple of (conversation_id, conversation_history)
        """
        conversation_id = self.start_conversation()
        
        for message in messages:
            try:
                self.query(message, conversation_id=conversation_id, **kwargs)
            except Exception as e:
                logger.error(f"Error in multi-turn conversation for message '{message[:50]}...': {str(e)}")
                break
        
        return conversation_id, self.get_conversation_history(conversation_id)
    
    def batch_query(self, messages: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Send multiple queries to the chatbot API.
        
        Args:
            messages: List of messages to send
            **kwargs: Additional parameters to include in the requests
            
        Returns:
            List of response dictionaries
        """
        results = []
        
        for message in messages:
            try:
                response = self.query(message, **kwargs)
                results.append(response)
            except Exception as e:
                logger.error(f"Error in batch query for message '{message[:50]}...': {str(e)}")
                results.append({"error": str(e)})
        
        return results


def get_chatbot_client(
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    test_client=None
) -> ChatbotClient:
    """
    Factory function to create and return a ChatbotClient instance.
    
    Args:
        endpoint: Optional chatbot API endpoint URL
        api_key: Optional API key for authentication
        test_client: Optional test client for direct integration testing
        
    Returns:
        ChatbotClient instance
    """
    return ChatbotClient(endpoint=endpoint, api_key=api_key, test_client=test_client)


if __name__ == "__main__":
    # Example usage
    client = get_chatbot_client()
    
    try:
        # Test single query
        print("\n=== Single Query ===")
        response = client.query("Hello, how are you?")
        print("Full response:", response)
        
        response_text = client.extract_response(response)
        print("Response text:", response_text)
        
        # Test multi-turn conversation
        print("\n=== Multi-turn Conversation ===")
        conversation_id = client.start_conversation()
        
        # First turn
        response1 = client.query_and_extract("What is the capital of France?", conversation_id=conversation_id)
        print("Response 1:", response1)
        
        # Second turn
        response2 = client.query_and_extract("What is its population?", conversation_id=conversation_id)
        print("Response 2:", response2)
        
        # Third turn
        response3 = client.query_and_extract("Tell me about its famous landmarks", conversation_id=conversation_id)
        print("Response 3:", response3)
        
        # Get conversation history
        history = client.get_conversation_history(conversation_id)
        print("\nConversation History:")
        for i, exchange in enumerate(history, 1):
            print(f"Turn {i}:")
            print(f"  User: {exchange['user']}")
            print(f"  Assistant: {exchange['assistant']}")
        
        # End conversation
        client.end_conversation(conversation_id)
    
    except Exception as e:
        print(f"Error: {str(e)}")
