import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { getHistory } from '../services/analysisService';
import { 
  History, 
  Search, 
  Filter, 
  ExternalLink, 
  Image as ImageIcon, 
  Video,
  Download
} from 'lucide-react';
import { motion } from 'framer-motion';

const ResultsHistory = () => {
    const { data: history = [], isLoading } = useQuery({
        queryKey: ['history'],
        queryFn: getHistory
    });

    return (
        <div className="max-w-7xl mx-auto space-y-8">
            <header className="flex justify-between items-end">
                <div>
                    <h1 className="text-3xl font-heading font-bold text-gray-900">Analysis History</h1>
                    <p className="text-gray-500 mt-1">Review all your previous authenticity reports.</p>
                </div>
            </header>

            <div className="bg-white rounded-3xl border border-gray-100 shadow-sm overflow-hidden min-h-[500px]">
                <div className="p-6 border-b border-gray-50 flex flex-col sm:flex-row gap-4 justify-between items-center">
                    <div className="relative w-full sm:w-96">
                        <Search size={18} className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-400" />
                        <input 
                            type="text" 
                            placeholder="Search by filename..." 
                            className="w-full bg-gray-50 pl-12 pr-4 py-2.5 rounded-xl text-sm border-none focus:ring-2 focus:ring-primary-500/20 outline-none"
                        />
                    </div>
                    <div className="flex gap-3 w-full sm:w-auto">
                        <button className="flex-1 sm:flex-none flex items-center justify-center gap-2 px-4 py-2.5 bg-gray-50 text-gray-700 text-sm font-bold rounded-xl hover:bg-gray-100 transition-colors">
                            <Filter size={18} /> Filters
                        </button>
                        <button className="flex-1 sm:flex-none flex items-center justify-center gap-2 px-4 py-2.5 bg-gray-50 text-gray-700 text-sm font-bold rounded-xl hover:bg-gray-100 transition-colors">
                            <Download size={18} /> Export CSV
                        </button>
                    </div>
                </div>

                <div className="overflow-x-auto">
                    <table className="w-full text-left">
                        <thead>
                            <tr className="bg-gray-50/30">
                                <th className="px-8 py-5 text-xs font-bold text-gray-400 uppercase tracking-widest">Media</th>
                                <th className="px-8 py-5 text-xs font-bold text-gray-400 uppercase tracking-widest">Verdict</th>
                                <th className="px-8 py-5 text-xs font-bold text-gray-400 uppercase tracking-widest">Confidence</th>
                                <th className="px-8 py-5 text-xs font-bold text-gray-400 uppercase tracking-widest">Date Analyzed</th>
                                <th className="px-8 py-5 text-xs font-bold text-gray-400 uppercase tracking-widest">Action</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-50">
                            {history.length > 0 ? (
                                history.map((item: any, i: number) => (
                                    <motion.tr 
                                        initial={{ opacity: 0, x: -10 }}
                                        animate={{ opacity: 1, x: 0 }}
                                        transition={{ delay: i * 0.05 }}
                                        key={item.id} 
                                        className="group hover:bg-primary-50/30 transition-colors"
                                    >
                                        <td className="px-8 py-5">
                                            <div className="flex items-center gap-4">
                                                <div className={`w-12 h-12 rounded-2xl flex items-center justify-center ${
                                                    item.type === 'image' ? 'bg-indigo-50 text-indigo-500' : 'bg-orange-50 text-orange-500'
                                                }`}>
                                                    {item.type === 'image' ? <ImageIcon size={24} /> : <Video size={24} />}
                                                </div>
                                                <div>
                                                    <p className="font-bold text-gray-900 group-hover:text-primary-600 transition-colors">{item.filename}</p>
                                                    <p className="text-xs text-gray-400 font-bold uppercase tracking-wider">{item.type}</p>
                                                </div>
                                            </div>
                                        </td>
                                        <td className="px-8 py-5 text-sm">
                                            <span className={`px-4 py-1.5 rounded-full text-xs font-bold ${
                                                item.score < 50 ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
                                            }`}>
                                                {item.verdict}
                                            </span>
                                        </td>
                                        <td className="px-8 py-5">
                                            <div className="flex items-center gap-2">
                                                <div className="w-16 h-1.5 bg-gray-100 rounded-full overflow-hidden">
                                                    <div 
                                                        className={`h-full rounded-full ${item.score > 50 ? 'bg-red-500' : 'bg-green-500'}`} 
                                                        style={{width: `${item.score}%`}}
                                                    ></div>
                                                </div>
                                                <span className="text-sm font-bold text-gray-900">{item.score}%</span>
                                            </div>
                                        </td>
                                        <td className="px-8 py-5 text-sm text-gray-500 font-medium">
                                            {new Date(item.timestamp).toLocaleDateString('en-US', {
                                                month: 'short',
                                                day: 'numeric',
                                                year: 'numeric',
                                                hour: '2-digit',
                                                minute: '2-digit'
                                            })}
                                        </td>
                                        <td className="px-8 py-5">
                                            <button className="p-2 text-gray-400 hover:text-primary-600 hover:bg-primary-50 rounded-lg transition-all">
                                                <ExternalLink size={20} />
                                            </button>
                                        </td>
                                    </motion.tr>
                                ))
                            ) : (
                                !isLoading && (
                                    <tr>
                                        <td colSpan={5} className="py-24 text-center">
                                            <div className="w-20 h-20 bg-gray-50 text-gray-300 rounded-3xl flex items-center justify-center mx-auto mb-6">
                                                <History size={40} />
                                            </div>
                                            <h3 className="text-xl font-bold font-heading text-gray-500 mb-2">No history found</h3>
                                            <p className="text-gray-400 text-sm max-w-xs mx-auto">Upload content to see your analysis results appear here.</p>
                                        </td>
                                    </tr>
                                )
                            )}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
};

export default ResultsHistory;
