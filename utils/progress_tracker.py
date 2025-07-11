
import time
import streamlit as st
from typing import Dict, Any, Optional
import logging

class ProgressTracker:
    """Utility for tracking and displaying processing progress."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.start_time = None
        self.total_steps = 0
        self.current_step = 0
        self.step_descriptions = {}
        self.progress_bar = None
        self.status_text = None
    
    def initialize(self, total_steps: int, step_descriptions: Dict[int, str] = None):
        """Initialize progress tracking."""
        self.total_steps = total_steps
        self.current_step = 0
        self.step_descriptions = step_descriptions or {}
        self.start_time = time.time()
        
        # Create Streamlit components
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
    
    def update_step(self, step: int, description: str = None):
        """Update current step."""
        self.current_step = step
        
        if description:
            self.step_descriptions[step] = description
        
        # Calculate progress percentage
        progress = min(step / self.total_steps, 1.0)
        
        # Update progress bar
        if self.progress_bar:
            self.progress_bar.progress(progress)
        
        # Update status text
        if self.status_text:
            current_desc = self.step_descriptions.get(step, f"Step {step}")
            elapsed_time = time.time() - self.start_time if self.start_time else 0
            
            # Estimate remaining time
            if step > 0:
                avg_time_per_step = elapsed_time / step
                remaining_steps = self.total_steps - step
                estimated_remaining = avg_time_per_step * remaining_steps
                
                status = f"{current_desc} ({step}/{self.total_steps}) - "
                status += f"Elapsed: {self._format_time(elapsed_time)} - "
                status += f"ETA: {self._format_time(estimated_remaining)}"
            else:
                status = f"{current_desc} ({step}/{self.total_steps})"
            
            self.status_text.text(status)
    
    def complete(self):
        """Mark progress as complete."""
        if self.progress_bar:
            self.progress_bar.progress(1.0)
        
        if self.status_text:
            elapsed_time = time.time() - self.start_time if self.start_time else 0
            self.status_text.text(f"✅ Completed in {self._format_time(elapsed_time)}")
    
    def _format_time(self, seconds: float) -> str:
        """Format time in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds // 60
            seconds = seconds % 60
            return f"{int(minutes)}m {int(seconds)}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{int(hours)}h {int(minutes)}m"
    
    def get_progress_info(self) -> Dict[str, Any]:
        """Get current progress information."""
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        return {
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'progress_percentage': (self.current_step / self.total_steps) * 100,
            'elapsed_time': elapsed_time,
            'current_description': self.step_descriptions.get(self.current_step, ''),
            'is_complete': self.current_step >= self.total_steps
        }

class BatchProgressTracker:
    """Progress tracker for batch processing operations."""
    
    def __init__(self, batch_size: int = 10):
        self.batch_size = batch_size
        self.total_items = 0
        self.processed_items = 0
        self.current_batch = 0
        self.start_time = None
        self.batch_start_time = None
        
        # Streamlit components
        self.main_progress = None
        self.batch_progress = None
        self.status_text = None
    
    def initialize(self, total_items: int):
        """Initialize batch processing."""
        self.total_items = total_items
        self.processed_items = 0
        self.current_batch = 0
        self.start_time = time.time()
        
        # Create Streamlit components
        st.subheader("📊 Processing Progress")
        
        col1, col2 = st.columns(2)
        with col1:
            st.text("Overall Progress")
            self.main_progress = st.progress(0)
        
        with col2:
            st.text("Current Batch")
            self.batch_progress = st.progress(0)
        
        self.status_text = st.empty()
    
    def start_batch(self, batch_number: int):
        """Start processing a new batch."""
        self.current_batch = batch_number
        self.batch_start_time = time.time()
        
        if self.batch_progress:
            self.batch_progress.progress(0)
        
        self._update_status()
    
    def update_batch_progress(self, items_in_batch: int, current_item: int):
        """Update progress within current batch."""
        if self.batch_progress:
            batch_progress = current_item / items_in_batch
            self.batch_progress.progress(batch_progress)
        
        self._update_status()
    
    def complete_batch(self, items_processed: int):
        """Complete current batch."""
        self.processed_items += items_processed
        
        if self.main_progress:
            overall_progress = self.processed_items / self.total_items
            self.main_progress.progress(overall_progress)
        
        if self.batch_progress:
            self.batch_progress.progress(1.0)
        
        self._update_status()
    
    def _update_status(self):
        """Update status text."""
        if not self.status_text:
            return
        
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        # Calculate ETA
        if self.processed_items > 0:
            avg_time_per_item = elapsed_time / self.processed_items
            remaining_items = self.total_items - self.processed_items
            estimated_remaining = avg_time_per_item * remaining_items
            
            status = f"Processing batch {self.current_batch} - "
            status += f"{self.processed_items}/{self.total_items} items - "
            status += f"Elapsed: {self._format_time(elapsed_time)} - "
            status += f"ETA: {self._format_time(estimated_remaining)}"
        else:
            status = f"Starting batch {self.current_batch}..."
        
        self.status_text.text(status)
    
    def _format_time(self, seconds: float) -> str:
        """Format time in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds // 60
            seconds = seconds % 60
            return f"{int(minutes)}m {int(seconds)}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{int(hours)}h {int(minutes)}m"
