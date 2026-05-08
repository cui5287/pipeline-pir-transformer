"""
Alert System

Real-time alert generation and management system.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum
import json
from collections import deque


logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "ACTIVE"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"


class Alert:
    """Single alert object."""
    
    def __init__(self,
                 alert_id: str,
                 severity: AlertSeverity,
                 message: str,
                 pir_value: float,
                 timestamp: datetime = None,
                 location: str = "Pipeline Section A",
                 recommendations: List[str] = None):
        """
        Initialize alert.
        
        Args:
            alert_id: Unique alert identifier
            severity: Alert severity level
            message: Alert message
            pir_value: PIR value that triggered alert
            timestamp: Alert timestamp
            location: Pipeline location
            recommendations: Recommended actions
        """
        self.alert_id = alert_id
        self.severity = severity
        self.message = message
        self.pir_value = pir_value
        self.timestamp = timestamp or datetime.now()
        self.location = location
        self.recommendations = recommendations or []
        self.status = AlertStatus.ACTIVE
        self.acknowledged_at = None
        self.acknowledged_by = None
        self.resolution_notes = None
    
    def to_dict(self) -> Dict:
        """Convert alert to dictionary."""
        return {
            'alert_id': self.alert_id,
            'severity': self.severity.value,
            'message': self.message,
            'pir_value': float(self.pir_value),
            'timestamp': self.timestamp.isoformat(),
            'location': self.location,
            'recommendations': self.recommendations,
            'status': self.status.value,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'acknowledged_by': self.acknowledged_by,
            'resolution_notes': self.resolution_notes
        }
    
    def acknowledge(self, user: str, notes: str = "") -> None:
        """Acknowledge alert."""
        self.status = AlertStatus.ACKNOWLEDGED
        self.acknowledged_at = datetime.now()
        self.acknowledged_by = user
        self.resolution_notes = notes
        logger.info(f"Alert {self.alert_id} acknowledged by {user}")
    
    def resolve(self, notes: str = "") -> None:
        """Resolve alert."""
        self.status = AlertStatus.RESOLVED
        self.resolution_notes = notes
        logger.info(f"Alert {self.alert_id} resolved")


class AlertSystem:
    """Main alert management system."""
    
    # PIR thresholds for alert generation
    ALERT_THRESHOLDS = {
        AlertSeverity.LOW: (0.0, 0.3),
        AlertSeverity.MEDIUM: (0.3, 0.6),
        AlertSeverity.HIGH: (0.6, 0.8),
        AlertSeverity.CRITICAL: (0.8, 1.0)
    }
    
    # Recommendations by severity
    RECOMMENDATIONS = {
        AlertSeverity.LOW: [
            "Continue routine monitoring",
            "Record baseline measurements",
            "Schedule next inspection"
        ],
        AlertSeverity.MEDIUM: [
            "Increase monitoring frequency to 2 hours",
            "Prepare contingency plans",
            "Alert engineering team",
            "Review recent operational changes"
        ],
        AlertSeverity.HIGH: [
            "Increase monitoring frequency to 30 minutes",
            "Activate on-call response team",
            "Prepare for potential intervention",
            "Notify management and regulatory bodies",
            "Review pipeline stress analysis"
        ],
        AlertSeverity.CRITICAL: [
            "🚨 IMMEDIATE ACTION REQUIRED",
            "Emergency response team activation",
            "Consider controlled pressure reduction",
            "Prepare for pipeline section isolation",
            "Notify emergency services",
            "Activate business continuity plan"
        ]
    }
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize alert system.
        
        Args:
            max_history: Maximum number of alerts to keep in history
        """
        self.max_history = max_history
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history = deque(maxlen=max_history)
        self.alert_counter = 0
    
    def generate_alert(self,
                      pir_value: float,
                      confidence: float = None,
                      location: str = "Pipeline Section A") -> Optional[Alert]:
        """
        Generate alert based on PIR value.
        
        Args:
            pir_value: PIR prediction value (0-1)
            confidence: Prediction confidence
            location: Pipeline location
            
        Returns:
            Alert object if threshold exceeded, None otherwise
        """
        # Determine severity
        severity = self._determine_severity(pir_value)
        
        if severity is None:
            return None
        
        # Generate alert ID
        self.alert_counter += 1
        alert_id = f"ALERT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.alert_counter}"
        
        # Create alert message
        message = self._generate_message(severity, pir_value, confidence)
        
        # Get recommendations
        recommendations = self.RECOMMENDATIONS[severity].copy()
        
        # Create alert object
        alert = Alert(
            alert_id=alert_id,
            severity=severity,
            message=message,
            pir_value=pir_value,
            location=location,
            recommendations=recommendations
        )
        
        # Store active alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Log alert
        logger.warning(f"Alert generated: {alert_id} [{severity.value}] - {message}")
        
        return alert
    
    def _determine_severity(self, pir_value: float) -> Optional[AlertSeverity]:
        """Determine alert severity based on PIR value."""
        for severity, (lower, upper) in self.ALERT_THRESHOLDS.items():
            if lower <= pir_value < upper:
                return severity
        return None
    
    def _generate_message(self, severity: AlertSeverity,
                         pir_value: float,
                         confidence: float = None) -> str:
        """Generate alert message."""
        messages = {
            AlertSeverity.LOW: f"Low risk detected. PIR: {pir_value:.4f}",
            AlertSeverity.MEDIUM: f"Medium risk detected. PIR: {pir_value:.4f}. Increase monitoring.",
            AlertSeverity.HIGH: f"High risk detected. PIR: {pir_value:.4f}. Immediate attention required.",
            AlertSeverity.CRITICAL: f"CRITICAL RISK detected. PIR: {pir_value:.4f}. Emergency response required!"
        }
        
        message = messages.get(severity, "Unknown risk")
        
        if confidence is not None:
            message += f" (Confidence: {confidence:.2%})"
        
        return message
    
    def acknowledge_alert(self, alert_id: str, user: str, notes: str = "") -> bool:
        """
        Acknowledge an active alert.
        
        Args:
            alert_id: Alert ID
            user: User acknowledging the alert
            notes: Acknowledgment notes
            
        Returns:
            True if successful, False otherwise
        """
        if alert_id not in self.active_alerts:
            logger.error(f"Alert {alert_id} not found")
            return False
        
        alert = self.active_alerts[alert_id]
        alert.acknowledge(user, notes)
        
        return True
    
    def resolve_alert(self, alert_id: str, notes: str = "") -> bool:
        """
        Resolve an alert.
        
        Args:
            alert_id: Alert ID
            notes: Resolution notes
            
        Returns:
            True if successful, False otherwise
        """
        if alert_id not in self.active_alerts:
            logger.error(f"Alert {alert_id} not found")
            return False
        
        alert = self.active_alerts[alert_id]
        alert.resolve(notes)
        
        # Remove from active alerts
        del self.active_alerts[alert_id]
        
        return True
    
    def get_active_alerts(self) -> List[Dict]:
        """Get all active alerts."""
        return [alert.to_dict() for alert in self.active_alerts.values()]
    
    def get_alert_summary(self) -> Dict:
        """Get alert summary statistics."""
        active_by_severity = {}
        for severity in AlertSeverity:
            active_by_severity[severity.value] = 0
        
        for alert in self.active_alerts.values():
            active_by_severity[alert.severity.value] += 1
        
        return {
            'total_active': len(self.active_alerts),
            'total_history': len(self.alert_history),
            'active_by_severity': active_by_severity,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_alert_history(self, limit: int = 100) -> List[Dict]:
        """Get alert history."""
        history_list = list(self.alert_history)
        return [alert.to_dict() for alert in history_list[-limit:]]
    
    def export_alerts_json(self, filepath: str) -> None:
        """Export all alerts to JSON file."""
        data = {
            'timestamp': datetime.now().isoformat(),
            'summary': self.get_alert_summary(),
            'active_alerts': self.get_active_alerts(),
            'alert_history': self.get_alert_history(limit=500)
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Alerts exported to {filepath}")
    
    def clear_resolved_alerts(self) -> int:
        """Clear resolved alerts from history."""
        initial_size = len(self.alert_history)
        
        # The deque automatically manages maxlen, but we can clear old resolved alerts
        resolved_count = sum(1 for alert in self.alert_history if alert.status == AlertStatus.RESOLVED)
        
        logger.info(f"Cleared {resolved_count} resolved alerts")
        return resolved_count
    
    def get_alert_statistics(self) -> Dict:
        """Get detailed alert statistics."""
        total_alerts = len(self.alert_history)
        
        severity_stats = {severity.value: 0 for severity in AlertSeverity}
        status_stats = {status.value: 0 for status in AlertStatus}
        
        for alert in self.alert_history:
            severity_stats[alert.severity.value] += 1
            status_stats[alert.status.value] += 1
        
        # Calculate average PIR values by severity
        pir_by_severity = {severity.value: [] for severity in AlertSeverity}
        for alert in self.alert_history:
            pir_by_severity[alert.severity.value].append(alert.pir_value)
        
        avg_pir_by_severity = {
            severity: sum(values) / len(values) if values else 0
            for severity, values in pir_by_severity.items()
        }
        
        return {
            'total_alerts': total_alerts,
            'total_active': len(self.active_alerts),
            'severity_distribution': severity_stats,
            'status_distribution': status_stats,
            'avg_pir_by_severity': avg_pir_by_severity,
            'timestamp': datetime.now().isoformat()
        }


# Global alert system instance
_alert_system = None


def get_alert_system(max_history: int = 1000) -> AlertSystem:
    """Get or create global alert system."""
    global _alert_system
    if _alert_system is None:
        _alert_system = AlertSystem(max_history=max_history)
    return _alert_system


def reset_alert_system() -> None:
    """Reset global alert system."""
    global _alert_system
    _alert_system = None
