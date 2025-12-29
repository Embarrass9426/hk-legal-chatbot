import React from 'react';
import { ExternalLink, BookOpen, Scale } from 'lucide-react';

const ReferenceCard = ({ reference }) => {
  const { title, citation, source_url, type } = reference;

  return (
    <div className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg p-4 shadow-sm hover:shadow-md transition-shadow mb-3">
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-2 mb-2">
          {type === 'Ordinance' ? (
            <BookOpen className="w-4 h-4 text-blue-600 dark:text-blue-400" />
          ) : (
            <Scale className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
          )}
          <span className={`text-xs font-semibold px-2 py-0.5 rounded ${
            type === 'Ordinance' 
              ? 'bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300' 
              : 'bg-emerald-50 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300'
          }`}>
            {type}
          </span>
        </div>
        <a 
          href={source_url} 
          target="_blank" 
          rel="noopener noreferrer"
          className="text-slate-400 hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
        >
          <ExternalLink className="w-4 h-4" />
        </a>
      </div>
      <h4 className="text-sm font-bold text-slate-800 dark:text-slate-100 mb-1">{title}</h4>
      <p className="text-xs text-slate-500 dark:text-slate-400 font-mono">{citation}</p>
    </div>
  );
};

export default ReferenceCard;
