'use client';

import React, { useState, useEffect } from 'react';
import { Calendar, TrendingUp, Loader2, RefreshCw } from 'lucide-react';

interface EconomicEvent {
  event: string;
  country: string;
  actual: number | null;
  estimate: number | null;
  prev: number | null;
  time: string;
  impact: string;
}

interface EconomicCalendarProps {
  inlineMobile?: boolean;
}

export default function EconomicCalendar({ inlineMobile }: EconomicCalendarProps) {
  const [events, setEvents] = useState<EconomicEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const fetchEvents = async () => {
    try {
      setLoading(true);
      setError('');
      const response = await fetch('/api/economic-calendar');
      const data = await response.json();

      if (data.success) {
        // Filter for high impact US events and limit to 20
        const highImpact = data.events
          .filter((e: EconomicEvent) => e.impact === 'high' && e.country === 'US')
          .slice(0, 20);
        setEvents(highImpact);
      } else {
        setError(data.error || 'Failed to fetch economic calendar');
      }
    } catch (err: any) {
      setError(err.message || 'Network error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchEvents();

    // Auto-refresh every hour
    const interval = setInterval(() => {
      fetchEvents();
    }, 60 * 60 * 1000);

    return () => clearInterval(interval);
  }, []);

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  };

  const formatTime = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
  };

  const getImpactColor = (impact: string) => {
    switch (impact) {
      case 'high':
        return 'var(--danger)';
      case 'medium':
        return 'var(--warning)';
      case 'low':
        return 'var(--success)';
      default:
        return 'var(--text-4)';
    }
  };

  const upcomingEvents = events.filter(e => new Date(e.time) > new Date());
  const pastEvents = events.filter(e => new Date(e.time) <= new Date());

  return (
    <div className={`card ${inlineMobile ? 'w-full' : 'w-80'}`}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Calendar className="w-5 h-5" style={{ color: 'var(--accent)' }} />
          <span className="card-label">Economic Calendar</span>
        </div>

        <button
          onClick={() => fetchEvents()}
          disabled={loading}
          className="p-2 transition-all border disabled:opacity-50"
          style={{
            background: 'var(--bg-3)',
            borderColor: 'var(--bg-1)',
            color: 'var(--text-3)',
          }}
          title="Refresh data"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {error && (
        <div className="mb-4 p-3 border-2" style={{
          background: 'var(--bg-2)',
          borderColor: 'var(--danger)',
          color: 'var(--danger)'
        }}>
          <p className="text-sm">{error}</p>
        </div>
      )}

      {loading ? (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="w-8 h-8 animate-spin" style={{ color: 'var(--accent)' }} />
        </div>
      ) : (
        <>
          {/* Summary */}
          <div className="mb-4 p-3 border" style={{ background: 'var(--bg-3)', borderColor: 'var(--bg-1)' }}>
            <div className="text-xs mb-1" style={{ color: 'var(--text-4)' }}>High Impact US Events</div>
            <div className="text-2xl font-bold" style={{ color: 'var(--text-2)' }}>
              {upcomingEvents.length}
            </div>
            <div className="text-xs" style={{ color: 'var(--text-5)' }}>upcoming</div>
          </div>

          {/* Upcoming Events */}
          {upcomingEvents.length > 0 && (
            <div className="mb-4">
              <div className="text-xs font-semibold mb-2" style={{ color: 'var(--text-4)' }}>
                Upcoming Events
              </div>
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {upcomingEvents.map((event, index) => (
                  <div
                    key={index}
                    className="p-3 border"
                    style={{
                      background: 'var(--bg-2)',
                      borderColor: 'var(--bg-1)',
                      borderLeftWidth: '3px',
                      borderLeftColor: getImpactColor(event.impact),
                    }}
                  >
                    <div className="flex items-start justify-between gap-2 mb-2">
                      <div className="flex-1 min-w-0">
                        <div className="font-semibold text-sm" style={{ color: 'var(--text-2)' }}>
                          {event.event}
                        </div>
                        <div className="text-xs" style={{ color: 'var(--text-5)' }}>
                          {formatDate(event.time)} • {formatTime(event.time)} ET
                        </div>
                      </div>
                      <div
                        className="px-2 py-1 text-xs font-semibold uppercase"
                        style={{
                          background: getImpactColor(event.impact) + '20',
                          color: getImpactColor(event.impact),
                        }}
                      >
                        {event.impact}
                      </div>
                    </div>

                    {event.estimate !== null && (
                      <div className="text-xs">
                        <span style={{ color: 'var(--text-4)' }}>Estimate: </span>
                        <span className="font-mono font-semibold" style={{ color: 'var(--text-2)' }}>
                          {event.estimate}
                        </span>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Recent Past Events */}
          {pastEvents.length > 0 && (
            <div>
              <div className="text-xs font-semibold mb-2" style={{ color: 'var(--text-4)' }}>
                Recent Results
              </div>
              <div className="space-y-2 max-h-48 overflow-y-auto">
                {pastEvents.slice(0, 5).map((event, index) => (
                  <div
                    key={index}
                    className="p-3 border opacity-75"
                    style={{
                      background: 'var(--bg-2)',
                      borderColor: 'var(--bg-1)',
                    }}
                  >
                    <div className="font-semibold text-sm mb-1" style={{ color: 'var(--text-3)' }}>
                      {event.event}
                    </div>
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      {event.actual !== null && (
                        <div>
                          <span style={{ color: 'var(--text-4)' }}>Actual: </span>
                          <span className="font-mono font-semibold" style={{ color: 'var(--text-2)' }}>
                            {event.actual}
                          </span>
                        </div>
                      )}
                      {event.estimate !== null && (
                        <div>
                          <span style={{ color: 'var(--text-4)' }}>Est: </span>
                          <span className="font-mono" style={{ color: 'var(--text-3)' }}>
                            {event.estimate}
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {events.length === 0 && !loading && (
            <div className="text-center py-8" style={{ color: 'var(--text-4)' }}>
              <p className="text-sm">No high impact events found</p>
            </div>
          )}
        </>
      )}
    </div>
  );
}
