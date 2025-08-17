"""Scheduler for daily trading runs at 00:00 UTC"""
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime
import asyncio
import logging
from typing import Dict, Optional, Callable

from config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingScheduler:
    """Manages scheduled trading runs"""
    
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.daily_run_callback: Optional[Callable] = None
        self.is_running = False
        
    def set_daily_run_callback(self, callback: Callable):
        """Set the callback function for daily runs
        
        Args:
            callback: Async function to call at scheduled time
        """
        self.daily_run_callback = callback
        
    async def _daily_run_wrapper(self):
        """Wrapper for daily run with error handling"""
        try:
            logger.info(f"Starting scheduled daily run at {datetime.utcnow()}")
            
            if self.daily_run_callback:
                # Run for each configured symbol
                for symbol in config.SYMBOLS:
                    logger.info(f"Running daily pipeline for {symbol}")
                    await self.daily_run_callback(symbol)
                    
            logger.info("Daily run completed successfully")
            
        except Exception as e:
            logger.error(f"Error in daily run: {str(e)}", exc_info=True)
    
    def start(self):
        """Start the scheduler"""
        if not self.is_running:
            # Schedule daily run at 00:00 UTC
            self.scheduler.add_job(
                self._daily_run_wrapper,
                CronTrigger(
                    hour=config.DAILY_RUN_TIME.hour,
                    minute=config.DAILY_RUN_TIME.minute,
                    second=0
                ),
                id='daily_trading_run',
                replace_existing=True
            )
            
            self.scheduler.start()
            self.is_running = True
            logger.info(f"Scheduler started. Daily runs at {config.DAILY_RUN_TIME} UTC")
    
    def stop(self):
        """Stop the scheduler"""
        if self.is_running:
            self.scheduler.shutdown(wait=False)
            self.is_running = False
            logger.info("Scheduler stopped")
    
    def get_next_run_time(self) -> Optional[datetime]:
        """Get the next scheduled run time"""
        job = self.scheduler.get_job('daily_trading_run')
        if job:
            return job.next_run_time
        return None
    
    def get_status(self) -> Dict[str, any]:
        """Get scheduler status"""
        next_run = self.get_next_run_time()
        
        return {
            'is_running': self.is_running,
            'next_run_time': next_run.isoformat() if next_run else None,
            'scheduled_time': f"{config.DAILY_RUN_TIME} UTC",
            'symbols': config.SYMBOLS
        }
    
    async def trigger_manual_run(self, symbol: str) -> Dict[str, any]:
        """Trigger a manual run outside of schedule
        
        Args:
            symbol: Trading symbol to run
            
        Returns:
            Run results
        """
        logger.info(f"Manual run triggered for {symbol}")
        
        if self.daily_run_callback:
            try:
                result = await self.daily_run_callback(symbol)
                return {
                    'status': 'success',
                    'symbol': symbol,
                    'timestamp': datetime.utcnow().isoformat(),
                    'result': result
                }
            except Exception as e:
                logger.error(f"Error in manual run: {str(e)}", exc_info=True)
                return {
                    'status': 'error',
                    'symbol': symbol,
                    'timestamp': datetime.utcnow().isoformat(),
                    'error': str(e)
                }
        else:
            return {
                'status': 'error',
                'message': 'No callback configured'
            }

# Global scheduler instance
trading_scheduler = TradingScheduler()