import React from 'react';
import { useAuthStore } from '../../store/authStore';
import { Bell, Search, User as UserIcon } from 'lucide-react';

const Navbar = () => {
  const user = useAuthStore((state) => state.user);

  return (
    <header className="h-16 bg-white/80 backdrop-blur-md border-b border-gray-100 flex items-center justify-between px-8 sticky top-0 z-10">
      <div className="flex items-center gap-4 bg-gray-50 px-4 py-2 rounded-xl w-96">
        <Search size={18} className="text-gray-400" />
        <input 
          type="text" 
          placeholder="Search analysis results..." 
          className="bg-transparent border-none outline-none text-sm w-full"
        />
      </div>

      <div className="flex items-center gap-6">
        <button className="relative p-2 text-gray-500 hover:text-gray-900 transition-colors">
          <Bell size={20} />
          <span className="absolute top-2 right-2 w-2 h-2 bg-red-500 rounded-full border-2 border-white"></span>
        </button>

        <div className="flex items-center gap-3 pl-4 border-l border-gray-100">
          <div className="text-right hidden sm:block">
            <p className="text-sm font-semibold text-gray-900">{user?.name || 'User'}</p>
            <p className="text-xs text-gray-500">{user?.email || 'user@example.com'}</p>
          </div>
          <div className="w-10 h-10 bg-gradient-to-tr from-primary-500 to-blue-600 rounded-full flex items-center justify-center text-white font-bold">
            {user?.name?.[0].toUpperCase() || <UserIcon size={20} />}
          </div>
        </div>
      </div>
    </header>
  );
};

export default Navbar;
