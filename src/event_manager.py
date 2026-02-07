"""
Event Manager
Manages exogenous events by date and provides anomaly levels

Author: ST-GCN Enhanced System
Date: Feb 2026

Responsibilities:
- Load exogenous events from JSON
- Query events by date
- Calculate daily anomaly levels
- Track event history
"""

import json
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import List, Dict, Tuple
import logging
import numpy as np

from .event_anomaly_detector import EventAnomalyDetector, load_default_detector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventManager:
    """
    Centralized event management
    
    Loads events from exogenous_events_geocoded.json and provides
    query methods for date-based event retrieval and anomaly calculation
    """
    
    def __init__(self, events_file: str = 'data/exogenous_events_geocoded.json',
                 detector: EventAnomalyDetector = None):
        """
        Initialize EventManager
        
        Args:
            events_file: Path to events JSON file
            detector: EventAnomalyDetector instance (defaults to built-in)
        """
        self.events_file = Path(events_file)
        self.detector = detector or load_default_detector()
        self.events = []
        self.event_index = {}  # Date -> list of event indices
        
        self._load_events()
    
    def _load_events(self):
        """Load events from JSON file"""
        try:
            if self.events_file.exists():
                with open(self.events_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Handle both direct list and wrapped object
                if isinstance(data, list):
                    self.events = data
                elif isinstance(data, dict) and 'events' in data:
                    self.events = data['events']
                else:
                    self.events = []
                
                # Build date index
                self._build_date_index()
                
                logger.info(f"Loaded {len(self.events)} events from {self.events_file}")
            else:
                logger.warning(f"Events file not found: {self.events_file}. Using empty events.")
                self.events = []
        except Exception as e:
            logger.error(f"Error loading events: {e}. Using empty events.")
            self.events = []
    
    def _build_date_index(self):
        """Build efficient date-based index of events"""
        self.event_index = {}
        
        for idx, event in enumerate(self.events):
            # Try multiple date field names
            event_date = None
            for date_field in ['date', 'event_date', 'datetime', 'created_at']:
                if date_field in event:
                    event_date = event[date_field]
                    break
            
            if event_date:
                # Parse date if it's a string
                if isinstance(event_date, str):
                    try:
                        # Try various date formats
                        for fmt in ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%d/%m/%Y']:
                            try:
                                event_date = datetime.strptime(event_date, fmt).date()
                                break
                            except ValueError:
                                continue
                    except:
                        continue
                
                # Ensure we have a date object
                if isinstance(event_date, datetime):
                    event_date = event_date.date()
                
                if isinstance(event_date, date):
                    event_date_str = str(event_date)
                    if event_date_str not in self.event_index:
                        self.event_index[event_date_str] = []
                    self.event_index[event_date_str].append(idx)
    
    def get_events_for_date(self, target_date: date) -> List[Dict]:
        """
        Get all events for a specific date
        
        Args:
            target_date: datetime.date object
        
        Returns:
            List of event dictionaries for that date
        """
        if isinstance(target_date, datetime):
            target_date = target_date.date()
        
        date_str = str(target_date)
        indices = self.event_index.get(date_str, [])
        
        return [self.events[i] for i in indices if i < len(self.events)]
    
    def get_events_for_date_range(self, start_date: date, end_date: date) -> List[Dict]:
        """
        Get all events within a date range
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
        
        Returns:
            List of events in range
        """
        events_in_range = []
        current_date = start_date
        
        while current_date <= end_date:
            events_in_range.extend(self.get_events_for_date(current_date))
            current_date += timedelta(days=1)
        
        return events_in_range
    
    def get_anomaly_level_for_date(self, target_date: date) -> float:
        """
        Calculate anomaly level (0-1) for a specific date
        
        Returns the maximum severity among all events that day
        (One critical event makes the whole day high-risk)
        
        Args:
            target_date: datetime.date object
        
        Returns:
            Anomaly level in [0, 1]
        """
        events = self.get_events_for_date(target_date)
        
        if not events:
            return 0.0
        
        # Parse events and get severities
        event_texts = [
            e.get('text', '') or e.get('description', '') or e.get('event', '')
            for e in events
        ]
        parsed = self.detector.parse_events_batch(event_texts)
        
        # Return max severity
        if parsed:
            severities = [p['severity'] for p in parsed]
            return max(severities)
        
        return 0.0
    
    def get_anomaly_level_for_range(self, start_date: date, end_date: date) -> Dict[str, float]:
        """
        Calculate anomaly levels for each date in range
        
        Args:
            start_date: Start date
            end_date: End date
        
        Returns:
            Dictionary mapping date -> anomaly_level
        """
        anomaly_levels = {}
        current_date = start_date
        
        while current_date <= end_date:
            anomaly_levels[str(current_date)] = self.get_anomaly_level_for_date(current_date)
            current_date += timedelta(days=1)
        
        return anomaly_levels
    
    def get_recent_events(self, days_back: int = 7) -> List[Dict]:
        """
        Get recent events (last N days)
        
        Args:
            days_back: How many days to look back
        
        Returns:
            List of recent events with details
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)
        
        events = self.get_events_for_date_range(start_date, end_date)
        
        # Enrich with severity info
        enriched = []
        for event in events:
            event_text = event.get('text', '') or event.get('description', '')
            parsed = self.detector.parse_event(event_text)
            
            enriched.append({
                'date': event.get('date', str(date.today())),
                'text': event_text,
                'location': event.get('location', event.get('bairro', 'Unknown')),
                'severity': parsed['severity'],
                'anomaly': parsed['anomaly_flag'],
                'crime_types': parsed['crime_types'],
                'original_event': event
            })
        
        return enriched
    
    def get_anomaly_warnings(self, days_back: int = 7) -> List[Dict]:
        """
        Get anomaly warnings (events with severity > 0.6)
        
        Args:
            days_back: Look back period
        
        Returns:
            List of high-severity events
        """
        recent = self.get_recent_events(days_back)
        return [e for e in recent if e['anomaly']]
    
    def get_statistics(self) -> Dict:
        """
        Get overall event statistics
        
        Returns:
            Dictionary with stats
        """
        if not self.events:
            return {
                'total_events': 0,
                'date_range': None,
                'events_by_date': {},
                'average_events_per_day': 0.0
            }
        
        # Find date range
        dates = []
        for date_str in self.event_index.keys():
            try:
                d = datetime.strptime(date_str, '%Y-%m-%d').date()
                dates.append(d)
            except:
                pass
        
        if not dates:
            return {'total_events': len(self.events), 'date_range': None}
        
        min_date = min(dates)
        max_date = max(dates)
        days_span = (max_date - min_date).days + 1
        
        return {
            'total_events': len(self.events),
            'date_range': f"{min_date} to {max_date}",
            'days_span': days_span,
            'events_by_date': {
                date_str: len(indices)
                for date_str, indices in self.event_index.items()
            },
            'average_events_per_day': len(self.events) / max(days_span, 1),
            'max_events_in_single_day': max(len(indices) for indices in self.event_index.values()) if self.event_index else 0
        }


# Example usage and testing
if __name__ == "__main__":
    manager = EventManager()
    
    print("Event Manager Tests")
    print("=" * 80)
    
    # Statistics
    stats = manager.get_statistics()
    print(f"\nEvent Statistics:")
    print(f"  Total events: {stats['total_events']}")
    print(f"  Date range: {stats.get('date_range', 'N/A')}")
    print(f"  Avg events/day: {stats.get('average_events_per_day', 0):.1f}")
    
    # Sample date
    test_date = date.today()
    events_today = manager.get_events_for_date(test_date)
    anomaly_today = manager.get_anomaly_level_for_date(test_date)
    
    print(f"\nToday ({test_date}):")
    print(f"  Events: {len(events_today)}")
    print(f"  Anomaly level: {anomaly_today:.3f}")
    
    # Recent events
    recent = manager.get_recent_events(days_back=3)
    print(f"\nRecent events (last 3 days):")
    for event in recent[:5]:  # Show first 5
        print(f"  [{event['date']}] {event['text'][:50]}... (severity: {event['severity']:.2f})")
    
    # Warnings
    warnings = manager.get_anomaly_warnings(days_back=7)
    print(f"\nAnomalies (last 7 days): {len(warnings)}")
    for warning in warnings[:3]:
        print(f"  WARNING: {warning['text'][:60]}... (severity: {warning['severity']:.2f})")
