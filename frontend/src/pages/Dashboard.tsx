import React from 'react';
import { motion } from 'framer-motion';
import { 
  Upload, 
  Video, 
  History, 
  TrendingUp, 
  CheckCircle2, 
  AlertTriangle,
  ArrowRight,
  Link2,
  Activity
} from 'lucide-react';
import { Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { getHistory } from '../services/analysisService';
import { 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell
} from 'recharts';

const Dashboard = () => {
  const { data: history = [] } = useQuery({
    queryKey: ['history'],
    queryFn: getHistory
  });

  const stats = [
    { label: 'Total Scans', value: history.length, icon: History, color: 'bg-blue-50 text-blue-600' },
    { label: 'Real Detected', value: history.filter((h: any) => h.score < 50).length, icon: CheckCircle2, color: 'bg-green-50 text-green-600' },
    { label: 'Fakes Flagged', value: history.filter((h: any) => h.score >= 50).length, icon: AlertTriangle, color: 'bg-red-50 text-red-600' },
    { label: 'Accuracy', value: '99.8%', icon: TrendingUp, color: 'bg-purple-50 text-purple-600' },
  ];

  const chartData = [
    { name: 'Mon', scans: 4 },
    { name: 'Tue', scans: 7 },
    { name: 'Wed', scans: 5 },
    { name: 'Thu', scans: 12 },
    { name: 'Fri', scans: 8 },
    { name: 'Sat', scans: 3 },
    { name: 'Sun', scans: history.length },
  ];

  const pieData = [
    { name: 'Real', value: history.filter((h: any) => h.score < 50).length || 1 },
    { name: 'Fake', value: history.filter((h: any) => h.score >= 50).length || 0 },
  ];

  const COLORS = ['#10b981', '#ef4444'];

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      <header className="flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-heading font-bold text-gray-900">Dashboard Overview</h1>
          <p className="text-gray-500 mt-1">Monitor and manage your content authenticity scans.</p>
        </div>
        <div className="flex gap-4">
          <Link to="/analyze-image" className="btn-primary flex items-center gap-2">
            <Upload size={18} /> New Image Scan
          </Link>
          <Link to="/analyze-video" className="btn-secondary flex items-center gap-2">
            <Video size={18} /> New Video Scan
          </Link>
          <Link to="/analyze-multimodal" className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-violet-600 to-purple-600 text-white rounded-xl font-semibold shadow-lg shadow-violet-200 hover:opacity-90 transition-all active:scale-95">
            <Activity size={18} /> Multimodal
          </Link>
        </div>
      </header>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat, idx) => (
          <motion.div
            key={idx}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: idx * 0.1 }}
            className="bg-white p-6 rounded-3xl border border-gray-100 shadow-sm"
          >
            <div className={`w-12 h-12 ${stat.color} rounded-2xl flex items-center justify-center mb-4`}>
              <stat.icon size={24} />
            </div>
            <p className="text-sm font-semibold text-gray-400 uppercase tracking-wider">{stat.label}</p>
            <p className="text-3xl font-bold font-heading mt-1">{stat.value}</p>
          </motion.div>
        ))}
      </div>

      <div className="grid lg:grid-cols-3 gap-8">
        {/* Main Chart */}
        <div className="lg:col-span-2 bg-white p-8 rounded-3xl border border-gray-100 shadow-sm">
          <div className="flex justify-between items-center mb-8">
            <h3 className="text-xl font-bold font-heading">Scan Activity</h3>
            <select className="bg-gray-50 border-none rounded-lg text-sm font-semibold px-4 py-2 outline-none">
                <option>Last 7 Days</option>
                <option>Last 30 Days</option>
            </select>
          </div>
          <div className="h-[300px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="colorScans" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#0ea5e9" stopOpacity={0.1}/>
                    <stop offset="95%" stopColor="#0ea5e9" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{fill: '#94a3b8', fontSize: 12}} dy={10} />
                <YAxis axisLine={false} tickLine={false} tick={{fill: '#94a3b8', fontSize: 12}} />
                <Tooltip 
                    contentStyle={{borderRadius: '16px', border: 'none', boxShadow: '0 10px 15px -3px rgba(0,0,0,0.1)'}}
                />
                <Area type="monotone" dataKey="scans" stroke="#0ea5e9" strokeWidth={3} fillOpacity={1} fill="url(#colorScans)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Breakdown */}
        <div className="bg-white p-8 rounded-3xl border border-gray-100 shadow-sm flex flex-col">
          <h3 className="text-xl font-bold font-heading mb-8">Verdict Distribution</h3>
          <div className="flex-1 min-h-[250px] relative">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
            <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
                <span className="text-3xl font-bold font-heading">{history.length}</span>
                <span className="text-xs font-bold text-gray-400 uppercase">Total Items</span>
            </div>
          </div>
          <div className="flex justify-center gap-6 mt-4">
              <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-green-500"></div>
                  <span className="text-sm font-medium text-gray-600">Authentic</span>
              </div>
              <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-red-500"></div>
                  <span className="text-sm font-medium text-gray-600">Manipulated</span>
              </div>
          </div>
        </div>
      </div>

      {/* Recent Activity Table */}
      <div className="bg-white rounded-3xl border border-gray-100 shadow-sm overflow-hidden">
        <div className="p-8 border-b border-gray-50 flex justify-between items-center">
          <h3 className="text-xl font-bold font-heading">Recent Scans</h3>
          <Link to="/history" className="text-primary-600 font-bold text-sm flex items-center gap-1 hover:gap-2 transition-all">
            View All History <ArrowRight size={16} />
          </Link>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left">
            <thead>
              <tr className="bg-gray-50/50">
                <th className="px-8 py-4 text-xs font-bold text-gray-400 uppercase tracking-widest">Filename</th>
                <th className="px-8 py-4 text-xs font-bold text-gray-400 uppercase tracking-widest">Type</th>
                <th className="px-8 py-4 text-xs font-bold text-gray-400 uppercase tracking-widest">Verdict</th>
                <th className="px-8 py-4 text-xs font-bold text-gray-400 uppercase tracking-widest">Score</th>
                <th className="px-8 py-4 text-xs font-bold text-gray-400 uppercase tracking-widest">Date</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-50">
              {history.slice(0, 5).map((item: any) => (
                <tr key={item.id} className="hover:bg-gray-50/50 transition-colors">
                  <td className="px-8 py-4">
                    <span className="font-semibold text-gray-900">{item.filename}</span>
                  </td>
                  <td className="px-8 py-4">
                    <span className="px-3 py-1 bg-gray-100 text-gray-600 rounded-lg text-xs font-bold uppercase tracking-wider">
                        {item.type}
                    </span>
                  </td>
                  <td className="px-8 py-4">
                    <span className={`px-3 py-1 rounded-lg text-xs font-bold ${
                      item.score < 50 ? 'bg-green-50 text-green-600' : 'bg-red-50 text-red-600'
                    }`}>
                      {item.verdict}
                    </span>
                  </td>
                  <td className="px-8 py-4">
                    <div className="flex items-center gap-2">
                        <div className="w-full bg-gray-100 h-1.5 w-16 rounded-full">
                            <div className={`h-full rounded-full ${item.score > 50 ? 'bg-red-500' : 'bg-green-500'}`} style={{width: `${item.score}%`}}></div>
                        </div>
                        <span className="font-bold text-gray-900">{item.score}%</span>
                    </div>
                  </td>
                  <td className="px-8 py-4 text-gray-500 text-sm">
                    {new Date(item.timestamp).toLocaleDateString()}
                  </td>
                </tr>
              ))}
              {history.length === 0 && (
                <tr>
                   <td colSpan={5} className="px-8 py-12 text-center text-gray-400 font-medium">
                       No recent scans found. Start by uploading an image or video!
                   </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
