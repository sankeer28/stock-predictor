'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Bell, BellOff, Plus, X, CheckCircle, AlertCircle, ChevronUp, ChevronDown } from 'lucide-react';

interface PriceAlert {
  id: string;
  symbol: string;
  type: 'above' | 'below';
  targetPrice: number;
  createdAt: number;
  triggered: boolean;
  triggeredAt?: number;
}

interface PriceAlertsProps {
  symbol: string;
  currentPrice: number;
}

const STORAGE_KEY = 'stockPriceAlerts';

function genId() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
}

export default function PriceAlerts({ symbol: currentSymbol, currentPrice }: PriceAlertsProps) {
  const [alerts, setAlerts] = useState<PriceAlert[]>([]);
  const [targetInput, setTargetInput] = useState('');
  const [alertType, setAlertType] = useState<'above' | 'below'>('above');
  const [permission, setPermission] = useState<NotificationPermission>('default');
  const [collapsed, setCollapsed] = useState(false);
  const prevPriceRef = useRef<number>(currentPrice);
  const alertsRef = useRef<PriceAlert[]>([]);

  useEffect(() => {
    alertsRef.current = alerts;
  }, [alerts]);

  // Load alerts from localStorage
  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) setAlerts(JSON.parse(raw));
    } catch {}
    if ('Notification' in window) {
      setPermission(Notification.permission);
    }
  }, []);

  // Persist alerts
  useEffect(() => {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(alerts)); } catch {}
  }, [alerts]);

  const requestNotificationPermission = async () => {
    if (!('Notification' in window)) return;
    const result = await Notification.requestPermission();
    setPermission(result);
  };

  const sendNotification = useCallback((alert: PriceAlert, price: number) => {
    const title = `${alert.symbol} Alert Triggered!`;
    const body = `Price ${alert.type === 'above' ? 'rose above' : 'fell below'} $${alert.targetPrice.toFixed(2)} — now at $${price.toFixed(2)}`;

    if ('Notification' in window && Notification.permission === 'granted') {
      new Notification(title, { body, icon: '/favicon.ico' });
    }
  }, []);

  // Check alerts when price changes
  useEffect(() => {
    if (!currentPrice || !alertsRef.current.length) return;

    const updatedAlerts = alertsRef.current.map(alert => {
      if (alert.triggered || alert.symbol !== currentSymbol) return alert;

      const triggered =
        (alert.type === 'above' && currentPrice >= alert.targetPrice) ||
        (alert.type === 'below' && currentPrice <= alert.targetPrice);

      if (triggered) {
        sendNotification(alert, currentPrice);
        return { ...alert, triggered: true, triggeredAt: Date.now() };
      }
      return alert;
    });

    const hasChanges = updatedAlerts.some((a, i) => a.triggered !== alertsRef.current[i].triggered);
    if (hasChanges) setAlerts(updatedAlerts);

    prevPriceRef.current = currentPrice;
  }, [currentPrice, currentSymbol, sendNotification]);

  const addAlert = () => {
    const price = parseFloat(targetInput);
    if (isNaN(price) || price <= 0) return;

    const newAlert: PriceAlert = {
      id: genId(),
      symbol: currentSymbol,
      type: alertType,
      targetPrice: price,
      createdAt: Date.now(),
      triggered: false,
    };

    setAlerts(prev => [newAlert, ...prev]);
    setTargetInput('');
  };

  const removeAlert = (id: string) => {
    setAlerts(prev => prev.filter(a => a.id !== id));
  };

  const clearTriggered = () => {
    setAlerts(prev => prev.filter(a => !a.triggered));
  };

  const activeAlerts = alerts.filter(a => !a.triggered);
  const triggeredAlerts = alerts.filter(a => a.triggered);

  return (
    <div className="card">
      <span className="card-label">Price Alerts</span>

      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-1.5">
          <Bell className="w-3.5 h-3.5" style={{ color: 'var(--accent)' }} />
          <span className="text-xs font-semibold" style={{ color: 'var(--text-3)' }}>
            {activeAlerts.length} active · {triggeredAlerts.length} fired
          </span>
        </div>
        <button
          onClick={() => setCollapsed(c => !c)}
          className="p-1 border transition-opacity hover:opacity-70"
          style={{ background: 'var(--bg-3)', borderColor: 'var(--bg-1)', color: 'var(--text-4)' }}
        >
          {collapsed ? <ChevronDown className="w-3 h-3" /> : <ChevronUp className="w-3 h-3" />}
        </button>
      </div>

      {!collapsed && (
        <>
          {/* Notification permission */}
          {permission !== 'granted' && (
            <div className="flex items-center gap-2 mb-3 p-2 border" style={{ borderColor: 'var(--warning)', background: 'rgba(234,179,8,0.06)' }}>
              <BellOff className="w-3.5 h-3.5 flex-shrink-0" style={{ color: 'var(--warning)' }} />
              <span className="text-[10px] flex-1" style={{ color: 'var(--text-4)' }}>
                {permission === 'denied' ? 'Notifications blocked in browser settings' : 'Enable browser notifications for alerts'}
              </span>
              {permission !== 'denied' && (
                <button
                  onClick={requestNotificationPermission}
                  className="px-2 py-1 border text-[9px] font-semibold flex-shrink-0"
                  style={{ borderColor: 'var(--warning)', color: 'var(--warning)', background: 'transparent' }}
                >
                  Enable
                </button>
              )}
            </div>
          )}

          {/* Add alert form */}
          <div className="mb-3 p-2 border" style={{ borderColor: 'var(--bg-1)', background: 'var(--bg-2)' }}>
            <div className="text-[10px] mb-2 font-semibold" style={{ color: 'var(--text-4)' }}>
              {currentSymbol} · Current: ${currentPrice.toFixed(2)}
            </div>
            <div className="flex gap-1 mb-1.5">
              <button
                onClick={() => setAlertType('above')}
                className="flex-1 py-1 text-[10px] font-bold border transition-all"
                style={{
                  background: alertType === 'above' ? 'rgba(34,197,94,0.15)' : 'transparent',
                  borderColor: alertType === 'above' ? 'var(--success)' : 'var(--bg-1)',
                  color: alertType === 'above' ? 'var(--success)' : 'var(--text-4)',
                }}
              >
                ↑ Above
              </button>
              <button
                onClick={() => setAlertType('below')}
                className="flex-1 py-1 text-[10px] font-bold border transition-all"
                style={{
                  background: alertType === 'below' ? 'rgba(239,68,68,0.15)' : 'transparent',
                  borderColor: alertType === 'below' ? 'var(--danger)' : 'var(--bg-1)',
                  color: alertType === 'below' ? 'var(--danger)' : 'var(--text-4)',
                }}
              >
                ↓ Below
              </button>
            </div>
            <div className="flex gap-1">
              <input
                value={targetInput}
                onChange={e => setTargetInput(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && addAlert()}
                placeholder="Target price…"
                type="number"
                step="0.01"
                min="0"
                className="flex-1 px-2 py-1.5 text-xs border"
                style={{
                  background: 'var(--bg-3)',
                  borderColor: 'var(--bg-1)',
                  color: 'var(--text-2)',
                  fontFamily: 'DM Mono, monospace',
                }}
              />
              <button
                onClick={addAlert}
                disabled={!targetInput || isNaN(parseFloat(targetInput))}
                className="px-2 py-1.5 border text-xs font-bold transition-opacity hover:opacity-80 disabled:opacity-40"
                style={{ background: 'var(--accent)', borderColor: 'var(--accent)', color: 'var(--text-0)' }}
              >
                <Plus className="w-3 h-3" />
              </button>
            </div>
          </div>

          {/* Active alerts */}
          {activeAlerts.length > 0 && (
            <div className="mb-2">
              <div className="text-[9px] uppercase tracking-widest mb-1" style={{ color: 'var(--text-5)' }}>Active</div>
              {activeAlerts.map(alert => (
                <div
                  key={alert.id}
                  className="flex items-center justify-between px-2 py-1.5 mb-0.5 border"
                  style={{ borderColor: 'var(--bg-1)', background: 'var(--bg-2)' }}
                >
                  <div className="flex items-center gap-1.5">
                    <Bell className="w-3 h-3 flex-shrink-0" style={{ color: alert.type === 'above' ? 'var(--success)' : 'var(--danger)' }} />
                    <div>
                      <span className="text-[10px] font-mono font-bold" style={{ color: 'var(--text-2)' }}>
                        {alert.symbol}
                      </span>
                      <span className="text-[10px] mx-1" style={{ color: 'var(--text-4)' }}>
                        {alert.type === 'above' ? '↑' : '↓'}
                      </span>
                      <span className="text-[10px] font-mono font-semibold" style={{ color: alert.type === 'above' ? 'var(--success)' : 'var(--danger)' }}>
                        ${alert.targetPrice.toFixed(2)}
                      </span>
                    </div>
                  </div>
                  <button onClick={() => removeAlert(alert.id)} className="hover:opacity-70" style={{ color: 'var(--text-5)' }}>
                    <X className="w-3 h-3" />
                  </button>
                </div>
              ))}
            </div>
          )}

          {/* Triggered alerts */}
          {triggeredAlerts.length > 0 && (
            <div>
              <div className="flex items-center justify-between mb-1">
                <div className="text-[9px] uppercase tracking-widest" style={{ color: 'var(--text-5)' }}>Triggered</div>
                <button onClick={clearTriggered} className="text-[9px] hover:opacity-70" style={{ color: 'var(--danger)' }}>Clear all</button>
              </div>
              {triggeredAlerts.map(alert => (
                <div
                  key={alert.id}
                  className="flex items-center justify-between px-2 py-1.5 mb-0.5 border"
                  style={{ borderColor: 'var(--bg-1)', background: 'var(--bg-3)', opacity: 0.7 }}
                >
                  <div className="flex items-center gap-1.5">
                    <CheckCircle className="w-3 h-3 flex-shrink-0" style={{ color: 'var(--success)' }} />
                    <span className="text-[10px] font-mono line-through" style={{ color: 'var(--text-4)' }}>
                      {alert.symbol} {alert.type === 'above' ? '↑' : '↓'} ${alert.targetPrice.toFixed(2)}
                    </span>
                  </div>
                  <button onClick={() => removeAlert(alert.id)} className="hover:opacity-70" style={{ color: 'var(--text-5)' }}>
                    <X className="w-3 h-3" />
                  </button>
                </div>
              ))}
            </div>
          )}

          {alerts.length === 0 && (
            <p className="text-[10px] text-center py-2" style={{ color: 'var(--text-5)' }}>
              No alerts set. Add one above.
            </p>
          )}
        </>
      )}
    </div>
  );
}
