import pytest
import time
import httpx
import os
from unittest.mock import MagicMock, patch
from openai import AzureOpenAI, BadRequestError

# Path to the batch processor module
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.absolute()))
from batch_processor import BatchProcessor, BatchJob

@pytest.fixture
def mock_azure_client():
    """Fixture to create a mocked AzureOpenAI client."""
    with patch('openai.AzureOpenAI', autospec=True) as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        yield mock_client

@pytest.fixture
def batch_processor(mock_azure_client):
    """Fixture to create a BatchProcessor with mocked dependencies."""
    with patch('batch_processor.AzureOpenAI', return_value=mock_azure_client):
        processor = BatchProcessor(
            verification_batches_dir="test_batches",
            max_retries=3,
            azure_endpoint="https://test-endpoint.azure.com",
            api_key="test-key",
            split="test_split"
        )
        return processor

class TestBatchProcessor:
    """Test cases for the BatchProcessor with timeout handling."""

    def test_client_initialization_with_timeout(self):
        """Test that the client is initialized with appropriate timeout settings."""
        with patch('batch_processor.AzureOpenAI', autospec=True) as mock_client_cls:
            processor = BatchProcessor(
                verification_batches_dir="test_batches",
                max_retries=3,
                azure_endpoint="https://test-endpoint.azure.com",
                api_key="test-key",
                split="test_split"
            )
            
            # Verify that the AzureOpenAI client was initialized with timeout and retries
            mock_client_cls.assert_called_once()
            args, kwargs = mock_client_cls.call_args
            
            # Check that timeout is provided in kwargs
            assert 'timeout' in kwargs
            assert kwargs['timeout'].read == 120.0  # Check read timeout is 120 seconds
            
            # Check that max_retries is provided
            assert 'max_retries' in kwargs
            assert kwargs['max_retries'] == 3

    def test_api_timeout_recovery(self, batch_processor, mock_azure_client):
        """Test that the processor can recover from API timeouts."""
        # Set up a job
        job = BatchJob(input_file="test_file.jsonl")
        job.batch_id = "test-batch-id"
        
        # Mock the batches.retrieve method to simulate a timeout and then success
        mock_response = MagicMock()
        mock_response.status = "in_progress"
        
        # Configure mock to raise timeout exception once, then succeed
        mock_azure_client.batches.retrieve.side_effect = [
            httpx.ReadTimeout("Connection timed out"),  # First call times out
            mock_response,                             # Second call succeeds
        ]
        
        # Call check_batch_completion twice
        result1 = batch_processor.check_batch_completion(job)
        result2 = batch_processor.check_batch_completion(job)
        
        # Assert expectations
        assert result1 is False  # First call fails but doesn't crash
        assert result2 is False  # Second call succeeds and returns False for in_progress
        assert job.retries == 1  # Job should have one retry recorded
        assert job.status == "batch_in_progress"  # Job status should be updated

    def test_heartbeat_logging(self, batch_processor, mock_azure_client):
        """Test that heartbeat logs are generated during processing."""
        # Set up a mock logger to capture logs
        mock_logger = MagicMock()
        batch_processor.logger = mock_logger
        
        # Set up mocked file discovery
        with patch.object(batch_processor, 'discover_input_files', return_value=["test_file.jsonl"]):
            # Set up mocked batch submission and completion
            with patch.object(batch_processor, 'submit_single_batch') as mock_submit:
                job = BatchJob(input_file="test_file.jsonl")
                job.batch_id = "test-batch-id"
                job.status = "in_progress"
                mock_submit.return_value = job
                
                # Set up mocked batch completion check
                with patch.object(batch_processor, 'check_batch_completion', side_effect=[False, True]):
                    # Run with a very short check interval for testing
                    batch_processor.process_all_batches(check_interval_minutes=0.001)
        
        # Check that heartbeat logs were generated
        heartbeat_logs = [call for call in mock_logger.info.call_args_list if "Heartbeat at" in str(call)]
        assert len(heartbeat_logs) > 0, "Heartbeat logs should be generated"

    def test_exception_handling_in_monitoring_loop(self, batch_processor, mock_azure_client):
        """Test that exceptions in the monitoring loop are handled properly."""
        # Set up a mock logger to capture logs
        mock_logger = MagicMock()
        batch_processor.logger = mock_logger
        
        # Set up mocked file discovery
        with patch.object(batch_processor, 'discover_input_files', return_value=["test_file.jsonl"]):
            # Set up mocked batch submission
            with patch.object(batch_processor, 'submit_single_batch') as mock_submit:
                job = BatchJob(input_file="test_file.jsonl")
                job.batch_id = "test-batch-id"
                job.status = "in_progress"
                mock_submit.return_value = job
                
                # Set up mocked batch completion check to raise an exception and then succeed
                check_completion_mock = MagicMock()
                check_completion_mock.side_effect = [
                    Exception("Test exception"),  # First call raises exception
                    True                          # Second call succeeds
                ]
                batch_processor.check_batch_completion = check_completion_mock
                
                # Run with a very short check interval for testing
                batch_processor.process_all_batches(check_interval_minutes=0.001)
        
        # Check that error was logged and process continued
        error_logs = [call for call in mock_logger.error.call_args_list if "Error in batch monitoring" in str(call)]
        assert len(error_logs) > 0, "Error in monitoring loop should be logged"
        
        # Check that processing completed successfully
        completion_logs = [call for call in mock_logger.info.call_args_list if "SEQUENTIAL PROCESSING COMPLETE" in str(call)]
        assert len(completion_logs) > 0, "Process should complete despite exceptions"

    def test_max_retries_exceeded(self):
        """Test that a job is marked as failed after exceeding max retries."""
        # Mock the AzureOpenAI client
        with patch('batch_processor.AzureOpenAI', autospec=True) as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            
            # Create processor with max_retries=2
            processor = BatchProcessor(
                verification_batches_dir="test_batches",
                max_retries=2,
                azure_endpoint="https://test-endpoint.azure.com",
                api_key="test-key",
                split="test_split"
            )
            
            # Set up a job
            job = BatchJob(input_file="test_file.jsonl")
            job.batch_id = "test-batch-id"
            
            # Mock the batches.retrieve method to consistently raise exceptions
            mock_client.batches.retrieve.side_effect = Exception("API Error")
            
            # Reset the retry counter before the test
            job.retries = 0
            
            # First call - should fail but not exceed max retries
            result1 = processor.check_batch_completion(job)
            assert result1 is False
            assert job.retries == 1
            
            # Second call - should fail and mark the job as max retries exceeded
            result2 = processor.check_batch_completion(job)
            assert result2 is True  # Returns True when max retries exceeded
            assert job.retries == 2
            assert job.status == "max_retries_exceeded"
            assert job in processor.failed_jobs
            
            # We don't need a third call since the job is already marked as failed


if __name__ == "__main__":
    # Run pytest directly
    import sys
    pytest.main(["-v", __file__]) 