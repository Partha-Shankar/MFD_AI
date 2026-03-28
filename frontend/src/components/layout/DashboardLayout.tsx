import React from 'react';
import { Navigate, Outlet } from 'react-router-dom';
import Sidebar from './Sidebar';
import Navbar from './Navbar';
import { useAuthStore } from '../../store/authStore';

const DashboardLayout = () => {
  const token = useAuthStore((state) => state.token);

  if (!token) {
    return <Navigate to="/login" replace />;
  }

  return (
    <div className="min-h-screen bg-[#f8fafc]">
      <Sidebar />
      <div className="pl-64 flex flex-col min-h-screen">
        <Navbar />
        <main className="flex-1 p-8">
          <Outlet />
        </main>
      </div>
    </div>
  );
};

export default DashboardLayout;
