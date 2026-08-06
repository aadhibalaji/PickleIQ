import React from 'react'

export default function Footer({ left, right }) {
  return (
    <footer className="pickleiq-footer">
      <span>{left}</span><span>{right}</span>
    </footer>
  )
}
