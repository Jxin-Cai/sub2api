<template>
  <BaseDialog
    :show="show"
    :title="t('admin.accounts.dedupTitle')"
    width="normal"
    @close="handleClose"
  >
    <div class="space-y-4">
      <div v-if="!previewed && !loading" class="text-sm text-gray-600 dark:text-dark-300">
        {{ t('admin.accounts.dedupHint') }}
      </div>

      <div v-if="loading" class="flex items-center justify-center py-8">
        <Icon name="refresh" size="lg" class="animate-spin text-gray-400" />
      </div>

      <div v-if="previewed && !loading">
        <div v-if="preview && preview.duplicates.length > 0" class="space-y-3">
          <div
            class="rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-700 dark:border-amber-800 dark:bg-amber-900/20 dark:text-amber-400"
          >
            {{ t('admin.accounts.dedupFoundSummary', { groups: preview.duplicates.length, total: preview.total_remove }) }}
          </div>

          <div class="max-h-72 overflow-auto rounded-lg border border-gray-200 dark:border-dark-700">
            <table class="w-full text-sm">
              <thead class="sticky top-0 bg-gray-50 dark:bg-dark-800">
                <tr>
                  <th class="px-3 py-2 text-left font-medium text-gray-700 dark:text-dark-300">{{ t('admin.accounts.dedupColName') }}</th>
                  <th class="px-3 py-2 text-right font-medium text-gray-700 dark:text-dark-300">{{ t('admin.accounts.dedupColCount') }}</th>
                  <th class="px-3 py-2 text-right font-medium text-gray-700 dark:text-dark-300">{{ t('admin.accounts.dedupColRemove') }}</th>
                </tr>
              </thead>
              <tbody class="divide-y divide-gray-100 dark:divide-dark-700">
                <tr v-for="group in preview.duplicates" :key="group.name">
                  <td class="px-3 py-2 text-gray-900 dark:text-white">{{ group.name }}</td>
                  <td class="px-3 py-2 text-right text-gray-600 dark:text-dark-300">{{ group.count }}</td>
                  <td class="px-3 py-2 text-right text-red-600 dark:text-red-400">{{ group.remove_ids.length }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <div v-else class="py-6 text-center text-sm text-gray-500 dark:text-dark-400">
          {{ t('admin.accounts.dedupNoDuplicates') }}
        </div>
      </div>

      <div v-if="result" class="rounded-lg border border-green-200 bg-green-50 p-3 text-sm text-green-700 dark:border-green-800 dark:bg-green-900/20 dark:text-green-400">
        {{ t('admin.accounts.dedupResult', { removed: result.removed }) }}
      </div>
    </div>

    <template #footer>
      <div class="flex justify-end gap-3">
        <button class="btn btn-secondary" type="button" :disabled="loading || executing" @click="handleClose">
          {{ t('common.cancel') }}
        </button>
        <button
          v-if="!previewed"
          class="btn btn-primary"
          type="button"
          :disabled="loading"
          @click="handlePreview"
        >
          {{ t('admin.accounts.dedupScan') }}
        </button>
        <button
          v-else-if="preview && preview.duplicates.length > 0 && !result"
          class="btn btn-danger"
          type="button"
          :disabled="executing"
          @click="handleExecute"
        >
          {{ executing ? t('admin.accounts.dedupExecuting') : t('admin.accounts.dedupConfirm') }}
        </button>
      </div>
    </template>
  </BaseDialog>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import BaseDialog from '@/components/common/BaseDialog.vue'
import Icon from '@/components/icons/Icon.vue'
import { adminAPI } from '@/api/admin'
import { useAppStore } from '@/stores/app'
import type { DedupPreviewResult, DedupExecuteResult } from '@/api/admin/accounts'

interface Props {
  show: boolean
}

interface Emits {
  (e: 'close'): void
  (e: 'deduped'): void
}

const props = defineProps<Props>()
const emit = defineEmits<Emits>()

const { t } = useI18n()
const appStore = useAppStore()

const loading = ref(false)
const executing = ref(false)
const previewed = ref(false)
const preview = ref<DedupPreviewResult | null>(null)
const result = ref<DedupExecuteResult | null>(null)

watch(
  () => props.show,
  (open) => {
    if (open) {
      loading.value = false
      executing.value = false
      previewed.value = false
      preview.value = null
      result.value = null
    }
  }
)

const handleClose = () => {
  if (loading.value || executing.value) return
  if (result.value) {
    emit('deduped')
  }
  emit('close')
}

const handlePreview = async () => {
  loading.value = true
  try {
    preview.value = await adminAPI.accounts.dedupPreview()
    previewed.value = true
  } catch (error: any) {
    appStore.showError(error?.message || t('admin.accounts.dedupFailed'))
  } finally {
    loading.value = false
  }
}

const handleExecute = async () => {
  executing.value = true
  try {
    result.value = await adminAPI.accounts.dedupExecute()
    appStore.showSuccess(t('admin.accounts.dedupResult', { removed: result.value.removed }))
  } catch (error: any) {
    appStore.showError(error?.message || t('admin.accounts.dedupFailed'))
  } finally {
    executing.value = false
  }
}
</script>
