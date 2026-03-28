import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useAuthStore } from '../../store/authStore';
import { 
  LayoutDashboard, 
  Image as ImageIcon, 
  Video, 
  Link2,
  History, 
  HelpCircle, 
  Info, 
  User, 
  LogOut,
  ChevronRight,
  Mic,
  Activity
} from 'lucide-react';
import { motion } from 'framer-motion';

const Sidebar = () => {
  const location = useLocation();
  const logout = useAuthStore((state) => state.logout);

  const menuItems = [
    { path: '/dashboard', icon: LayoutDashboard, label: 'Dashboard' },
    { path: '/analyze-image', icon: ImageIcon, label: 'Image Analysis' },
    { path: '/analyze-video', icon: Video, label: 'Video Analysis' },
    { path: '/analyze-audio', icon: Mic, label: 'Audio Analysis' },
    { path: '/analyze-link', icon: Link2, label: 'Link Analysis' },
    { path: '/analyze-multimodal', icon: Activity, label: 'Multimodal Analysis', badge: 'ADVANCED' },
    { path: '/how-it-works', icon: HelpCircle, label: 'How It Works' },
    { path: '/about', icon: Info, label: 'About Us' },
    { path: '/profile', icon: User, label: 'Profile' },
  ];

  return (
    <div className="w-64 h-screen bg-white border-r border-gray-100 flex flex-col fixed left-0 top-0 z-20">
      <div className="p-6">
        <Link to="/" className="flex items-center gap-2">
          <div className="w-8 h-8 bg-primary-600 rounded-lg flex items-center justify-center">
            <span className="text-white font-bold">M</span>
          </div>
          <span className="font-heading font-bold text-xl text-gray-900">MFD AI</span>
        </Link>
      </div>

      <nav className="flex-1 px-4 py-4 space-y-1">
        {menuItems.map((item) => {
          const isActive = location.pathname === item.path;
          return (
            <Link
              key={item.path}
              to={item.path}
              className={`flex items-center justify-between px-4 py-3 rounded-xl transition-all ${
                isActive 
                  ? 'bg-primary-50 text-primary-600' 
                  : 'text-gray-500 hover:bg-gray-50 hover:text-gray-900'
              }`}
            >
              <div className="flex items-center gap-3">
                <item.icon size={20} />
                <span className="font-medium">{item.label}</span>
              </div>
              <div className="flex items-center gap-2">
                {(item as any).badge && !isActive && (
                  <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-violet-100 text-violet-600 uppercase tracking-wide">
                    {(item as any).badge}
                  </span>
                )}
                {isActive && (
                  <motion.div layoutId="sidebar-active" className="w-1.5 h-1.5 rounded-full bg-primary-600" />
                )}
              </div>
            </Link>
          );
        })}
      </nav>

      <div className="p-4 border-t border-gray-100">
        <button
          onClick={logout}
          className="flex items-center gap-3 px-4 py-3 w-full text-left text-gray-500 hover:text-red-600 hover:bg-red-50 rounded-xl transition-all"
        >
          <LogOut size={20} />
          <span className="font-medium">Logout</span>
        </button>
      </div>
    </div>
  );
};

export default Sidebar;
