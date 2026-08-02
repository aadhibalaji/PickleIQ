import React from 'react'

// Placeholder landing page — structure only, no real content/design yet.
export default function Home({ videos, current, onOpenTesting }) {
  return (
    <div>
      <section style={{ padding: '64px 16px', textAlign: 'center' }}>
        <h1>Pickleball Rally Detection</h1>
        <p className="muted">Automatically find and analyze rallies in your pickleball clips.</p>
        <button className="blue" onClick={() => { console.log('CTA clicked'); onOpenTesting() }}>
          Get Started
        </button>
      </section>
      <div className="muted" style={{ textAlign: 'center' }}>
        {videos.length} clip{videos.length === 1 ? '' : 's'} loaded
        {current ? ` · currently viewing video ${current}` : ''}
      </div>
    </div>
  )
}
