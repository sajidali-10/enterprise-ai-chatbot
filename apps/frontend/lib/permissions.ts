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
  canManageUsers: boolean

  // Feedback
  canSubmitFeedback: boolean

  // Debug metadata visibility
  canViewDebugMetadata: boolean
}

/**
 * Normalized permission flags used across the frontend UI.
 * All keys are camelCase for consistent consumption by components.
 */
export type PermissionFlags = RolePermissions

/**
 * Normalize backend snake_case permissions or camelCase permissions
 * into a single PermissionFlags object.
 *
 * Falls back to role-derived permissions when no permission object is provided.
 */
export function normalizePermissions(
  permissions: unknown,
  role?: string
): PermissionFlags {
  if (!permissions || typeof permissions !== 'object') {
    return getPermissions(role)
  }

  const p = permissions as Record<string, boolean>

  // Detect snake_case backend format
  if ('can_use_general_chat' in p) {
    return {
      canUseGeneralChat: p.can_use_general_chat ?? false,
      canUseKnowledgeBase: p.can_use_knowledge_base ?? false,
      canUseDebug: p.can_use_debug ?? false,
      canViewDocuments: p.can_view_documents ?? false,
      canUploadDocuments: p.can_upload_documents ?? false,
      canReindexDocuments: p.can_reindex_documents ?? false,
      canDeleteDocuments: p.can_delete_documents ?? false,
      canAccessObservability: p.can_access_observability ?? false,
      canAccessEvaluations: p.can_access_evaluations ?? false,
      canManageUsers: p.can_manage_users ?? false,
      canSubmitFeedback: p.can_submit_feedback ?? false,
      canViewDebugMetadata: p.can_use_debug ?? false,
    }
  }

  // Assume camelCase (already normalized or role permissions)
  return {
    canUseGeneralChat: p.canUseGeneralChat ?? false,
    canUseKnowledgeBase: p.canUseKnowledgeBase ?? false,
    canUseDebug: p.canUseDebug ?? false,
    canViewDocuments: p.canViewDocuments ?? false,
    canUploadDocuments: p.canUploadDocuments ?? false,
    canReindexDocuments: p.canReindexDocuments ?? false,
    canDeleteDocuments: p.canDeleteDocuments ?? false,
    canAccessObservability: p.canAccessObservability ?? false,
    canAccessEvaluations: p.canAccessEvaluations ?? false,
    canManageUsers: p.canManageUsers ?? false,
    canSubmitFeedback: p.canSubmitFeedback ?? false,
    canViewDebugMetadata: p.canViewDebugMetadata ?? false,
  }
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
    canManageUsers: true,
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
    canManageUsers: false,
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
    canManageUsers: false,
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