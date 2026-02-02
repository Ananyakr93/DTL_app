import { useState } from 'react';
import { FileText, Download, Calendar, Filter, Table } from 'lucide-react';
import { useStore } from '../store';
import { fetchHistoricalData } from '../api';
import { exportToCSV, calculateStats } from '../utils';
import type { HistoricalDataPoint } from '../types';

export default function ReportsPage() {
    const { city, currentData, predictions, settings } = useStore();
    const isDarkMode = settings.isDarkMode;

    const [dateRange, setDateRange] = useState({ start: '', end: '' });
    const [selectedPollutants, setSelectedPollutants] = useState<string[]>(['pm2_5', 'pm10', 'no2', 'so2', 'co', 'o3']);
    const [includeForecasts, setIncludeForecasts] = useState(true);
    const [includeHealth, setIncludeHealth] = useState(true);
    const [isGenerating, setIsGenerating] = useState(false);
    const [historicalData, setHistoricalData] = useState<HistoricalDataPoint[]>([]);

    // Toggle pollutant selection
    const togglePollutant = (pollutant: string) => {
        setSelectedPollutants((prev) =>
            prev.includes(pollutant) ? prev.filter((p) => p !== pollutant) : [...prev, pollutant]
        );
    };

    // Generate PDF report (using simple HTML to print method)
    const generatePDFReport = async () => {
        setIsGenerating(true);

        // Fetch historical data if date range is set
        let data = historicalData;
        if (dateRange.start && dateRange.end) {
            const days = Math.ceil((new Date(dateRange.end).getTime() - new Date(dateRange.start).getTime()) / (1000 * 60 * 60 * 24));
            data = await fetchHistoricalData(city, Math.max(days, 7));
            setHistoricalData(data);
        } else {
            data = await fetchHistoricalData(city, 30);
            setHistoricalData(data);
        }

        const stats = calculateStats(data);

        // Create printable HTML content
        const printContent = `
      <!DOCTYPE html>
      <html>
      <head>
        <title>AeroClean Air Quality Report - ${city}</title>
        <style>
          body { font-family: Arial, sans-serif; padding: 40px; color: #333; }
          h1 { color: #1c1f26; border-bottom: 3px solid #a3ff12; padding-bottom: 10px; }
          h2 { color: #2a2e38; margin-top: 30px; }
          .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px; }
          .logo { font-size: 24px; font-weight: bold; color: #a3ff12; }
          .date { color: #666; font-size: 14px; }
          .summary-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0; }
          .summary-card { background: #f5f5f5; padding: 20px; border-radius: 10px; text-align: center; }
          .summary-card .value { font-size: 32px; font-weight: bold; color: #1c1f26; }
          .summary-card .label { font-size: 12px; color: #666; margin-top: 5px; }
          table { width: 100%; border-collapse: collapse; margin-top: 20px; }
          th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
          th { background: #f0f0f0; font-weight: bold; }
          .good { color: #22c55e; }
          .moderate { color: #eab308; }
          .poor { color: #f97316; }
          .severe { color: #ef4444; }
          .health-section { background: #f9fafb; padding: 20px; border-radius: 10px; margin-top: 20px; }
          .forecast-section { margin-top: 30px; }
          @media print { body { padding: 20px; } }
        </style>
      </head>
      <body>
        <div class="header">
          <div class="logo">🌿 AeroClean</div>
          <div class="date">Generated: ${new Date().toLocaleDateString('en-IN', { dateStyle: 'full' })}</div>
        </div>
        
        <h1>Air Quality Report - ${city}</h1>
        
        <h2>Current Conditions</h2>
        <div class="summary-grid">
          <div class="summary-card">
            <div class="value ${currentData?.current.aqi_value ? (currentData.current.aqi_value <= 100 ? 'good' : currentData.current.aqi_value <= 200 ? 'moderate' : 'severe') : ''}">${currentData?.current.aqi_value || 'N/A'}</div>
            <div class="label">Current AQI</div>
          </div>
          <div class="summary-card">
            <div class="value">${currentData?.current.aqi_status || 'N/A'}</div>
            <div class="label">Status</div>
          </div>
          <div class="summary-card">
            <div class="value">${currentData?.current.dominant_pollutant || 'PM2.5'}</div>
            <div class="label">Dominant Pollutant</div>
          </div>
          <div class="summary-card">
            <div class="value">${currentData?.current.station || city}</div>
            <div class="label">Station</div>
          </div>
        </div>

        <h2>Historical Summary (Last 30 Days)</h2>
        <div class="summary-grid">
          <div class="summary-card">
            <div class="value">${stats.avgAqi}</div>
            <div class="label">Average AQI</div>
          </div>
          <div class="summary-card">
            <div class="value severe">${stats.maxAqi}</div>
            <div class="label">Maximum AQI</div>
          </div>
          <div class="summary-card">
            <div class="value good">${stats.minAqi}</div>
            <div class="label">Minimum AQI</div>
          </div>
          <div class="summary-card">
            <div class="value">${stats.trend}</div>
            <div class="label">Trend</div>
          </div>
        </div>

        <h2>Pollutant Levels</h2>
        <table>
          <tr>
            <th>Pollutant</th>
            <th>Current Value</th>
            <th>Unit</th>
            <th>30-Day Average</th>
          </tr>
          ${selectedPollutants.includes('pm2_5') ? `
          <tr>
            <td>PM2.5</td>
            <td>${currentData?.current.pm2_5 ?? 'N/A'}</td>
            <td>µg/m³</td>
            <td>${data.length ? Math.round(data.reduce((a, b) => a + b.pm2_5, 0) / data.length) : 'N/A'}</td>
          </tr>` : ''}
          ${selectedPollutants.includes('pm10') ? `
          <tr>
            <td>PM10</td>
            <td>${currentData?.current.pm10 ?? 'N/A'}</td>
            <td>µg/m³</td>
            <td>${data.length ? Math.round(data.reduce((a, b) => a + b.pm10, 0) / data.length) : 'N/A'}</td>
          </tr>` : ''}
          ${selectedPollutants.includes('no2') ? `
          <tr>
            <td>NO₂</td>
            <td>${currentData?.current.no2 ?? 'N/A'}</td>
            <td>µg/m³</td>
            <td>${data.length ? Math.round(data.reduce((a, b) => a + b.no2, 0) / data.length) : 'N/A'}</td>
          </tr>` : ''}
          ${selectedPollutants.includes('so2') ? `
          <tr>
            <td>SO₂</td>
            <td>${currentData?.current.so2 ?? 'N/A'}</td>
            <td>µg/m³</td>
            <td>${data.length ? Math.round(data.reduce((a, b) => a + b.so2, 0) / data.length) : 'N/A'}</td>
          </tr>` : ''}
          ${selectedPollutants.includes('co') ? `
          <tr>
            <td>CO</td>
            <td>${currentData?.current.co ?? 'N/A'}</td>
            <td>mg/m³</td>
            <td>${data.length ? (data.reduce((a, b) => a + b.co, 0) / data.length).toFixed(1) : 'N/A'}</td>
          </tr>` : ''}
          ${selectedPollutants.includes('o3') ? `
          <tr>
            <td>O₃</td>
            <td>${currentData?.current.o3 ?? 'N/A'}</td>
            <td>µg/m³</td>
            <td>${data.length ? Math.round(data.reduce((a, b) => a + b.o3, 0) / data.length) : 'N/A'}</td>
          </tr>` : ''}
        </table>

        ${includeHealth ? `
        <h2>Health Recommendations</h2>
        <div class="health-section">
          <p><strong>General:</strong> ${currentData?.health.general || 'No data available'}</p>
          <p><strong>For Sensitive Groups:</strong> ${currentData?.health.sensitive || 'No data available'}</p>
        </div>
        ` : ''}

        ${includeForecasts ? `
        <h2>24-Hour Forecast</h2>
        <table>
          <tr>
            <th>Time</th>
            <th>Predicted AQI</th>
            <th>Status</th>
          </tr>
          ${predictions.slice(0, 12).map((p) => `
          <tr>
            <td>${p.hour}</td>
            <td>${p.aqi}</td>
            <td>${p.status}</td>
          </tr>
          `).join('')}
        </table>
        ` : ''}

        <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; font-size: 12px; color: #666;">
          <p>Report generated by AeroClean Dashboard | Data sources: CPCB, WAQI</p>
          <p>This report is for informational purposes only. For health decisions, consult local authorities.</p>
        </div>
      </body>
      </html>
    `;

        // Open in new window and print
        const printWindow = window.open('', '_blank');
        if (printWindow) {
            printWindow.document.write(printContent);
            printWindow.document.close();
            setTimeout(() => {
                printWindow.print();
            }, 500);
        }

        setIsGenerating(false);
    };

    // Export CSV
    const handleExportCSV = async () => {
        setIsGenerating(true);
        const data = await fetchHistoricalData(city, 30);
        exportToCSV(data, `aeroclean-report-${city}-${new Date().toISOString().split('T')[0]}.csv`);
        setIsGenerating(false);
    };

    return (
        <div className="space-y-6 animate-fade-in">
            {/* Header */}
            <div>
                <h1 className={`text-2xl font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                    <FileText className="inline w-7 h-7 mr-2 text-brand-primary" />
                    Reports - {city}
                </h1>
                <p className={`text-sm mt-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                    Generate and download air quality reports in PDF or CSV format
                </p>
            </div>

            {/* Report Configuration */}
            <div className={`p-6 rounded-xl ${isDarkMode ? 'bg-slate-800' : 'bg-white'} shadow-lg`}>
                <h3 className={`text-lg font-semibold mb-6 flex items-center gap-2 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                    <Filter className="w-5 h-5" />
                    Report Configuration
                </h3>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* Date Range */}
                    <div>
                        <label className={`block text-sm font-medium mb-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                            <Calendar className="inline w-4 h-4 mr-1" />
                            Date Range (Optional)
                        </label>
                        <div className="flex gap-2">
                            <input
                                type="date"
                                value={dateRange.start}
                                onChange={(e) => setDateRange({ ...dateRange, start: e.target.value })}
                                className={`flex-1 px-3 py-2 rounded-lg border ${isDarkMode ? 'bg-slate-700 text-white border-slate-600' : 'bg-white text-gray-900 border-gray-200'
                                    } focus:outline-none focus:ring-2 focus:ring-brand-primary`}
                            />
                            <span className={`self-center ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>to</span>
                            <input
                                type="date"
                                value={dateRange.end}
                                onChange={(e) => setDateRange({ ...dateRange, end: e.target.value })}
                                className={`flex-1 px-3 py-2 rounded-lg border ${isDarkMode ? 'bg-slate-700 text-white border-slate-600' : 'bg-white text-gray-900 border-gray-200'
                                    } focus:outline-none focus:ring-2 focus:ring-brand-primary`}
                            />
                        </div>
                    </div>

                    {/* Pollutant Selection */}
                    <div>
                        <label className={`block text-sm font-medium mb-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                            Select Pollutants
                        </label>
                        <div className="flex flex-wrap gap-2">
                            {['pm2_5', 'pm10', 'no2', 'so2', 'co', 'o3'].map((pollutant) => (
                                <button
                                    key={pollutant}
                                    onClick={() => togglePollutant(pollutant)}
                                    className={`px-3 py-1 rounded-full text-sm font-medium transition-colors ${selectedPollutants.includes(pollutant)
                                        ? 'bg-brand-primary text-brand-dark'
                                        : isDarkMode
                                            ? 'bg-slate-700 text-gray-300'
                                            : 'bg-gray-100 text-gray-600'
                                        }`}
                                >
                                    {pollutant.toUpperCase().replace('_', '.')}
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* Include Options */}
                    <div>
                        <label className={`block text-sm font-medium mb-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                            Include in Report
                        </label>
                        <div className="space-y-2">
                            <label className="flex items-center gap-2 cursor-pointer">
                                <input
                                    type="checkbox"
                                    checked={includeForecasts}
                                    onChange={(e) => setIncludeForecasts(e.target.checked)}
                                    className="w-4 h-4 rounded border-gray-300 text-brand-primary focus:ring-brand-primary"
                                />
                                <span className={isDarkMode ? 'text-gray-300' : 'text-gray-700'}>24-Hour Forecasts</span>
                            </label>
                            <label className="flex items-center gap-2 cursor-pointer">
                                <input
                                    type="checkbox"
                                    checked={includeHealth}
                                    onChange={(e) => setIncludeHealth(e.target.checked)}
                                    className="w-4 h-4 rounded border-gray-300 text-brand-primary focus:ring-brand-primary"
                                />
                                <span className={isDarkMode ? 'text-gray-300' : 'text-gray-700'}>Health Recommendations</span>
                            </label>
                        </div>
                    </div>
                </div>
            </div>

            {/* Download Options */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* PDF Report */}
                <div className={`p-6 rounded-xl ${isDarkMode ? 'bg-slate-800' : 'bg-white'} shadow-lg`}>
                    <div className="flex items-start gap-4">
                        <div className="p-3 rounded-xl bg-red-100 dark:bg-red-900/30">
                            <FileText className="w-8 h-8 text-red-500" />
                        </div>
                        <div className="flex-1">
                            <h4 className={`font-semibold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>PDF Report</h4>
                            <p className={`text-sm mt-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                                Complete report with charts, tables, and health recommendations. Ready for printing or sharing.
                            </p>
                            <button
                                onClick={generatePDFReport}
                                disabled={isGenerating}
                                className="mt-4 flex items-center gap-2 px-4 py-2 bg-red-500 text-white rounded-lg font-medium hover:bg-red-600 transition-colors disabled:opacity-50"
                            >
                                <Download className="w-4 h-4" />
                                {isGenerating ? 'Generating...' : 'Generate PDF'}
                            </button>
                        </div>
                    </div>
                </div>

                {/* CSV Export */}
                <div className={`p-6 rounded-xl ${isDarkMode ? 'bg-slate-800' : 'bg-white'} shadow-lg`}>
                    <div className="flex items-start gap-4">
                        <div className="p-3 rounded-xl bg-green-100 dark:bg-green-900/30">
                            <Table className="w-8 h-8 text-green-500" />
                        </div>
                        <div className="flex-1">
                            <h4 className={`font-semibold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>CSV Export</h4>
                            <p className={`text-sm mt-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                                Raw data export for analysis in Excel, Google Sheets, or other tools.
                            </p>
                            <button
                                onClick={handleExportCSV}
                                disabled={isGenerating}
                                className="mt-4 flex items-center gap-2 px-4 py-2 bg-green-500 text-white rounded-lg font-medium hover:bg-green-600 transition-colors disabled:opacity-50"
                            >
                                <Download className="w-4 h-4" />
                                {isGenerating ? 'Exporting...' : 'Export CSV'}
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            {/* Current Data Preview */}
            <div className={`p-6 rounded-xl ${isDarkMode ? 'bg-slate-800' : 'bg-white'} shadow-lg`}>
                <h3 className={`text-lg font-semibold mb-4 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                    Current Data Preview
                </h3>
                <div className="overflow-x-auto">
                    <table className="w-full">
                        <thead>
                            <tr className={isDarkMode ? 'border-b border-slate-700' : 'border-b border-gray-200'}>
                                <th className={`py-3 px-4 text-left text-sm font-semibold ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>Field</th>
                                <th className={`py-3 px-4 text-left text-sm font-semibold ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>Value</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr className={isDarkMode ? 'border-b border-slate-700' : 'border-b border-gray-100'}>
                                <td className={`py-3 px-4 ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>City</td>
                                <td className={`py-3 px-4 font-medium ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>{city}</td>
                            </tr>
                            <tr className={isDarkMode ? 'border-b border-slate-700' : 'border-b border-gray-100'}>
                                <td className={`py-3 px-4 ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>Current AQI</td>
                                <td className={`py-3 px-4 font-bold text-lg`} style={{ color: currentData ? getAQIColor(currentData.current.aqi_value) : undefined }}>
                                    {currentData?.current.aqi_value || 'N/A'}
                                </td>
                            </tr>
                            <tr className={isDarkMode ? 'border-b border-slate-700' : 'border-b border-gray-100'}>
                                <td className={`py-3 px-4 ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>Status</td>
                                <td className={`py-3 px-4 font-medium ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>{currentData?.current.aqi_status || 'N/A'}</td>
                            </tr>
                            <tr className={isDarkMode ? 'border-b border-slate-700' : 'border-b border-gray-100'}>
                                <td className={`py-3 px-4 ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>Station</td>
                                <td className={`py-3 px-4 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>{currentData?.current.station || 'N/A'}</td>
                            </tr>
                            <tr>
                                <td className={`py-3 px-4 ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>Data Source</td>
                                <td className={`py-3 px-4 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>{currentData?.current.aqi_source || 'N/A'}</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
}

function getAQIColor(aqi: number): string {
    if (aqi <= 50) return '#22c55e';
    if (aqi <= 100) return '#84cc16';
    if (aqi <= 200) return '#eab308';
    if (aqi <= 300) return '#f97316';
    if (aqi <= 400) return '#ef4444';
    return '#9333ea';
}
