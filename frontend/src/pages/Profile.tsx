import React from 'react';
import { useAuthStore } from '../store/authStore';
import { User, Mail, Shield, Key, History, LogOut, ChevronRight } from 'lucide-react';
import { motion } from 'framer-motion';

const Profile = () => {
    const { user, logout } = useAuthStore();

    return (
        <div className="max-w-4xl mx-auto space-y-8">
            <header>
                <h1 className="text-3xl font-heading font-bold text-gray-900">Your Account</h1>
                <p className="text-gray-500 mt-1">Manage your professional AI detector profile.</p>
            </header>

            <div className="grid md:grid-cols-3 gap-8">
                {/* Left side info */}
                <div className="md:col-span-1 space-y-6">
                    <div className="bg-white p-8 rounded-3xl border border-gray-100 shadow-sm flex flex-col items-center">
                        <div className="w-24 h-24 bg-gradient-to-tr from-primary-500 to-blue-600 rounded-full flex items-center justify-center text-white text-3xl font-bold mb-4 shadow-lg shadow-primary-100">
                            {user?.name?.[0].toUpperCase()}
                        </div>
                        <h2 className="text-xl font-bold font-heading text-gray-900">{user?.name}</h2>
                        <p className="text-sm text-gray-500 mb-6">{user?.email}</p>
                        <button className="w-full btn-secondary text-sm py-2">Edit Avatar</button>
                    </div>

                    <div className="bg-white p-6 rounded-3xl border border-gray-100 shadow-sm">
                        <h3 className="text-sm font-bold text-gray-400 uppercase tracking-widest mb-6 px-2">Account Stats</h3>
                        <div className="space-y-4">
                            <div className="flex justify-between items-center px-2">
                                <span className="text-gray-500 text-sm">Member Since</span>
                                <span className="font-bold text-gray-900 text-sm">March 2026</span>
                            </div>
                            <div className="flex justify-between items-center px-2">
                                <span className="text-gray-500 text-sm">Plan</span>
                                <span className="px-2 py-0.5 bg-primary-50 text-primary-600 text-[10px] font-bold rounded uppercase">Professional</span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Right side settings */}
                <div className="md:col-span-2 space-y-8">
                    <div className="bg-white rounded-3xl border border-gray-100 shadow-sm overflow-hidden">
                        <div className="p-6 border-b border-gray-50">
                            <h3 className="font-heading font-bold text-lg">Personal Information</h3>
                        </div>
                        <div className="p-8 space-y-6">
                            <div className="grid sm:grid-cols-2 gap-6">
                                <div className="space-y-2">
                                    <label className="text-xs font-bold text-gray-400 uppercase px-1">Full Name</label>
                                    <div className="p-3 bg-gray-50 rounded-xl border border-gray-100 text-gray-900 font-medium">
                                        {user?.name}
                                    </div>
                                </div>
                                <div className="space-y-2">
                                    <label className="text-xs font-bold text-gray-400 uppercase px-1">Email</label>
                                    <div className="p-3 bg-gray-50 rounded-xl border border-gray-100 text-gray-900 font-medium">
                                        {user?.email}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div className="bg-white rounded-3xl border border-gray-100 shadow-sm overflow-hidden">
                        <div className="p-6 border-b border-gray-50">
                            <h3 className="font-heading font-bold text-lg">Security Settings</h3>
                        </div>
                        <div className="p-4">
                            {[
                                { icon: Key, title: 'Change Password', desc: 'Update your login credentials' },
                                { icon: Shield, title: 'Two-Factor Authentication', desc: 'Add an extra layer of security' },
                                { icon: History, title: 'Session History', desc: 'Manage your active devices and logins' }
                            ].map((item, i) => (
                                <button key={i} className="w-full flex items-center justify-between p-4 hover:bg-gray-50 rounded-2xl transition-colors group">
                                    <div className="flex items-center gap-4">
                                        <div className="w-10 h-10 bg-gray-50 text-gray-400 rounded-xl flex items-center justify-center group-hover:bg-white group-hover:text-primary-600 transition-colors shadow-sm">
                                            <item.icon size={20} />
                                        </div>
                                        <div className="text-left">
                                            <p className="font-bold text-gray-900 text-sm">{item.title}</p>
                                            <p className="text-xs text-gray-500">{item.desc}</p>
                                        </div>
                                    </div>
                                    <ChevronRight size={18} className="text-gray-300" />
                                </button>
                            ))}
                        </div>
                    </div>

                    <button 
                        onClick={logout}
                        className="w-full p-4 bg-red-50 text-red-600 rounded-3xl font-bold flex items-center justify-center gap-3 hover:bg-red-100 transition-colors border border-red-100"
                    >
                        <LogOut size={20} /> Sign Out of All Devices
                    </button>
                </div>
            </div>
        </div>
    );
};

export default Profile;
