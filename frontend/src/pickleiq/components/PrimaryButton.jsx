import React from 'react'

export default function PrimaryButton({ children, className = '', ...props }) {
  return (
    <button className={`pickleiq-primary-button ${className}`.trim()} {...props}>
      {children}
    </button>
  )
}
