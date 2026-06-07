import React from 'react';
import { ExternalLink, BookOpen, Scale } from 'lucide-react';

const truncateWords = (text, maxWords = 50) => {
  if (!text) return '';
  const words = text.split(/\s+/);
  if (words.length <= maxWords) return text;
  return words.slice(0, maxWords).join(' ') + '...';
};

const ReferenceCard = ({ reference }) => {
  const { title, citation, source_url, type, page } = reference;
  const displayTitle = truncateWords(title, 50);

  return (
    <a
      href={source_url}
      target="_blank"
      rel="noopener noreferrer"
      className="block bg-charcoal-600 border border-charcoal-400 rounded-xl p-4 transition-all duration-200 hover:-translate-y-0.5 hover:shadow-lg hover:shadow-gold-400/5 hover:border-gold-400/30 group"
    >
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-2 mb-2">
          {type === 'Ordinance' ? (
            <BookOpen className="w-4 h-4 text-gold-400" />
          ) : (
            <Scale className="w-4 h-4 text-gold-400" />
          )}
          <span className="text-xs font-semibold px-2 py-0.5 rounded bg-gold-400/10 text-gold-400 border border-gold-400/20">
            {type}
          </span>
        </div>
        <div className="flex items-center gap-1 text-muted-text group-hover:text-gold-400 transition-colors">
          {page && <span className="text-[10px] font-bold uppercase">Page {page}</span>}
          <ExternalLink className="w-4 h-4" />
        </div>
      </div>
      <h4 className="text-sm font-semibold text-primary-text mb-1 group-hover:text-gold-300 transition-colors">{displayTitle}</h4>
      <p className="text-xs text-muted-text font-mono">{citation}</p>
    </a>
  );
};

export default ReferenceCard;
