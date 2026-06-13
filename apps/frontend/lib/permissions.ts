/**
 * Dev-mode Role-Based Permissions
 * 
 * This module defines a permission map for development mode simulation.
 * Each role has a set of capabilities that control UI visibility and behavior.
 * 
 * Future Phase 12/12.1: Replace with backend-provided permissions structure:
 * 
 *   current_user = {
 *     id,
 *     username,
 *     email,
 *     role,
 *     permissions: {
 *       can_upload_documents,
 *       can_delete_documents,
 *       can_reindex_documents,
 *       can_access_observability,
 *       can_access_evaluations,
 *       can_use_debug,
 *       ...
 *     }
 *   }
 * 
 * The frontend should be designed to consume current_user.permissions
 * rather than hardcoded role checks.
 */

export type UserRole = 'admin' | 'user' | 'viewer'

export interface RolePermissions {
  // Chat capabilities
  canUseGeneralChat: boolean
  canUseKnowledgeBase: boolean
  canUseDebug: boolean
  
  // Document capabilities
  canViewDocuments: boolean
  canUploadDocuments: boolean
  canReindexDocuments: boolean
  canDeleteDocuments: boolean
  
  // Admin capabilities
  canAccessObservability: boolean
  canAccessEvaluations: boolean
  
  // Feedback
  canSubmitFeedback: boolean
  
  // Debug metadata visibility
  canViewDebugMetadata: boolean
}

export const rolePermissions: Record<UserRole, RolePermissions> = {
  admin: {
    canUseGeneralChat: true,
    canUseKnowledgeBase: true,
    canUseDebug: true,
    canViewDocuments: true,
    canUploadDocuments: true,
    canReindexDocuments: true,
    canDeleteDocuments: true,
    canAccessObservability: true,
    canAccessEvaluations: true,
    canSubmitFeedback: true,
    canViewDebugMetadata: true,
  },
  user: {
    canUseGeneralChat: true,
    canUseKnowledgeBase: true,
    canUseDebug: false,
    canViewDocuments: true,
    canUploadDocuments: true,
    canReindexDocuments: false,
    canDeleteDocuments: false,
    canAccessObservability: false,
    canAccessEvaluations: false,
    canSubmitFeedback: true,
    canViewDebugMetadata: false,
  },
  viewer: {
    canUseGeneralChat: false,
    canUseKnowledgeBase: true,
    canUseDebug: false,
    canViewDocuments: true,
    canUploadDocuments: false,
    canReindexDocuments: false,
    canDeleteDocuments: false,
    canAccessObservability: false,
    canAccessEvaluations: false,
    canSubmitFeedback: true,
    canViewDebugMetadata: false,
  },
}

/**
 * Get permissions for a given role.
 */
export function getPermissions(role: UserRole | string | undefined): RolePermissions {
  if (!role) {
    return rolePermissions.viewer // Default to most restrictive
  }
  return rolePermissions[role as UserRole] || rolePermissions.viewer
}

/**
 * Check if a role has a specific permission.
 */
export function hasPermission(
  role: UserRole | string | undefined,
  permission: keyof RolePermissions
): boolean {
  const perms = getPermissions(role)
  return perms[permission] as boolean
}

/**
 * Get the default chat mode for a role.
 */
export function getDefaultChatMode(role: UserRole | string | undefined): 'general_chat' | 'knowledge_base' {
  const perms = getPermissions(role)
  if (perms.canUseGeneralChat) {
    return 'general_chat'
  }
  return 'knowledge_base'
}